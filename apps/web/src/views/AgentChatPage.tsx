import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  ActionIcon,
  Avatar,
  Badge,
  Button,
  Card,
  FileButton,
  Group,
  Image,
  ScrollArea,
  Stack,
  Switch,
  Text,
  Textarea,
  Title,
} from "@mantine/core";
import { useMediaQuery } from "@mantine/hooks";
import { notifications } from "@mantine/notifications";
import { IconPhoto, IconSend, IconTrash } from "@tabler/icons-react";

import MarkdownBlocks from "../components/MarkdownBlocks";
import { useAuth } from "../state/auth";
import { useI18n } from "../state/i18n";
import { apiFetchJson, toBase64 } from "../utils/api";

type ChatMsg = {
  role: "user" | "assistant";
  content: string;
  imageName?: string;
  imagePreviewUrl?: string;
  resultImageUrl?: string;
  resultTitle?: string;
  resultSubtitle?: string;
  resultScore?: string;
};

function formatFileSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function resolveMediaUrl(imagePath: string | null | undefined): string | null {
  if (!imagePath) return null;
  const normalized = String(imagePath).trim();
  if (!normalized) return null;
  if (/^https?:\/\//i.test(normalized)) return normalized;

  const marker = "/media/";
  const idx = normalized.lastIndexOf(marker);
  const relative = idx >= 0 ? normalized.slice(idx + marker.length) : normalized.split("/media/").pop() || "";
  if (!relative) return null;

  const apiBase = ((import.meta as any).env?.VITE_API_BASE_URL || "").replace(/\/+$/, "");
  return `${apiBase}/media/${relative.replace(/^\/+/, "")}`;
}

function buildSearchResultCard(result: any): { imageUrl?: string; title?: string; subtitle?: string; score?: string } | null {
  if (!result || typeof result !== "object") return null;
  const images = Array.isArray(result.images) ? result.images : [];
  const firstImage = images[0] && typeof images[0] === "object" ? images[0] : null;
  const imageUrl = resolveMediaUrl(firstImage?.image_path) || undefined;
  const title = String(result.model_id || "").trim() || undefined;
  const subtitle =
    [result.maker, result.part_number, result.category]
      .map((value) => String(value || "").trim())
      .filter(Boolean)
      .join(" / ") || undefined;
  let score: string | undefined;
  if (result.score !== undefined && result.score !== null) {
    const num = Number(result.score);
    if (Number.isFinite(num)) score = num.toFixed(3);
  }
  if (!imageUrl && !title && !subtitle) return null;
  return { imageUrl, title, subtitle, score };
}

export default function AgentChatPage() {
  const auth = useAuth();
  const { language, t } = useI18n();
  const isMobile = useMediaQuery("(max-width: 48em)");
  const messagePreviewUrlsRef = useRef<string[]>([]);
  const [messages, setMessages] = useState<ChatMsg[]>([
    {
      role: "assistant",
      content: t("agent.intro"),
    },
  ]);
  const [input, setInput] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [selectedPreviewUrl, setSelectedPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [encodingPercent, setEncodingPercent] = useState<number | null>(null);
  const [updateMilvus, setUpdateMilvus] = useState(false);

  const canSend = useMemo(() => Boolean(input.trim() || imageFile), [input, imageFile]);

  useEffect(() => {
    if (!imageFile) {
      setSelectedPreviewUrl(null);
      return;
    }

    const objectUrl = URL.createObjectURL(imageFile);
    setSelectedPreviewUrl(objectUrl);

    return () => URL.revokeObjectURL(objectUrl);
  }, [imageFile]);

  useEffect(
    () => () => {
      messagePreviewUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
      messagePreviewUrlsRef.current = [];
    },
    [],
  );

  useEffect(() => {
    setMessages((prev) => {
      if (prev.length !== 1 || prev[0]?.role !== "assistant") return prev;
      return [{ ...prev[0], content: t("agent.intro") }];
    });
  }, [language, t]);

  async function send() {
    if (!canSend) return;
    const text = input.trim() || t("agent.defaultPrompt");
    const selected = imageFile;
    const userMessage: ChatMsg = { role: "user", content: text };

    if (selected) {
      const sentPreviewUrl = URL.createObjectURL(selected);
      messagePreviewUrlsRef.current.push(sentPreviewUrl);
      userMessage.imageName = selected.name;
      userMessage.imagePreviewUrl = sentPreviewUrl;
    }

    setInput("");
    setLoading(true);
    setMessages((prev) => [...prev, userMessage]);

    try {
      if (selected) {
        console.info("[AgentChat] preparing upload", {
          fileName: selected.name,
          fileSizeBytes: selected.size,
          fileType: selected.type,
        });
      }
      const image_base64 = selected
        ? await toBase64(selected, {
            maxBytes: 5 * 1024 * 1024,
            maxDimension: 1600,
            quality: 0.82,
            onProgress: (pct) => setEncodingPercent(pct),
          })
        : null;
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (auth.token) headers.Authorization = `Bearer ${auth.token}`;

      const res = await apiFetchJson<{ answer: string; debug?: any; sources?: any[]; identified?: any; search_results?: any[] }>(
        "/api/v1/agent/chat",
        {
        method: "POST",
        headers,
        body: JSON.stringify({
          message: text,
          image_base64,
          max_tool_results: 5,
          update_milvus: updateMilvus,
        }),
        },
      );

      const sources = Array.isArray((res as any).sources) ? (res as any).sources : [];
      const sourcesText =
        sources.length > 0
          ? `\n\nSources:\n${sources
              .slice(0, 5)
              .map((s: any) => `- ${String(s.title || "").trim()} ${String(s.url || "").trim()}`.trim())
              .join("\n")}`
          : "";

      const identified = res.identified && typeof res.identified === "object" ? res.identified : {};
      const identifiedImages = Array.isArray(identified.images) ? identified.images : [];
      const firstImage = identifiedImages[0] && typeof identifiedImages[0] === "object" ? identifiedImages[0] : null;
      const resultImageUrl = resolveMediaUrl(firstImage?.image_path);
      const resultTitle = String(identified.model_id || "").trim() || undefined;
      const resultSubtitle = [identified.maker, identified.part_number, identified.category]
        .map((value) => String(value || "").trim())
        .filter(Boolean)
        .join(" / ") || undefined;
      const rawSearchResults = Array.isArray((res as any).search_results) ? (res as any).search_results : [];
      const representativeResult =
        buildSearchResultCard(identified) ||
        rawSearchResults.map((item: any) => buildSearchResultCard(item)).find(Boolean) ||
        null;

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: (res.answer || "(empty)") + sourcesText,
          resultImageUrl: representativeResult?.imageUrl || resultImageUrl || undefined,
          resultTitle: representativeResult?.title || resultTitle,
          resultSubtitle: representativeResult?.subtitle || resultSubtitle,
          resultScore: representativeResult?.score,
        },
      ]);
      setImageFile(null);
      console.info("[AgentChat] request completed", {
        hasImage: Boolean(selected),
        answerLength: String(res.answer || "").length,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      notifications.show({ color: "red", title: t("agent.errorTitle"), message: msg });
      console.error("[AgentChat] request failed", { hasImage: Boolean(selected), error: msg });
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: t("agent.errorPrefix", { message: msg }) },
      ]);
    } finally {
      setLoading(false);
      setEncodingPercent(null);
    }
  }

  return (
    <Stack
      gap="md"
      style={{
        // Prefer dynamic viewport on mobile browsers; keep vh fallback via minHeight.
        minHeight: "calc(100vh - 110px)",
        height: "calc(100dvh - 110px)",
      }}
    >
      <Group justify="space-between" align="baseline">
        <Title order={3}>{t("agent.title")}</Title>
        <Badge variant="light">{t("agent.badge")}</Badge>
      </Group>

      <Card withBorder radius="lg" p="md" style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
        <ScrollArea style={{ flex: 1 }}>
          <Stack gap="sm">
            {messages.map((m, idx) => (
              <Group key={idx} align="flex-start" justify={m.role === "user" ? "flex-end" : "flex-start"}>
                {m.role === "assistant" ? <Avatar size="sm" radius="xl" /> : null}
                <Card
                  withBorder
                  radius="lg"
                  p="sm"
                  style={{
                    maxWidth: 720,
                    borderColor: "rgba(255,255,255,0.10)",
                    backgroundColor: m.role === "user" ? "rgba(76,110,245,0.18)" : "rgba(255,255,255,0.06)",
                  }}
                >
                  {m.imagePreviewUrl ? (
                    <Stack gap={6} mb={m.content ? "xs" : 0}>
                      <Image
                        src={m.imagePreviewUrl}
                        alt={m.imageName || t("agent.uploadedImageAlt")}
                        radius="md"
                        h={160}
                        w={220}
                        fit="cover"
                      />
                      {m.imageName ? (
                        <Text size="xs" c="dimmed">
                          {m.imageName}
                        </Text>
                      ) : null}
                    </Stack>
                  ) : null}
                  {m.resultImageUrl ? (
                    <Stack gap={6} mb={m.content ? "xs" : 0}>
                      <Image
                        src={m.resultImageUrl}
                        alt={m.resultTitle || t("agent.matchedProductAlt")}
                        radius="md"
                        h={160}
                        w={220}
                        fit="cover"
                      />
                      {m.resultTitle || m.resultSubtitle ? (
                        <Stack gap={0}>
                          {m.resultTitle ? (
                            <Text size="xs" fw={600}>
                              {m.resultTitle}
                            </Text>
                          ) : null}
                          {m.resultSubtitle ? (
                            <Text size="xs" c="dimmed">
                              {m.resultSubtitle}
                            </Text>
                          ) : null}
                          {m.resultScore ? <Badge variant="light">{t("search.scoreBadge", { score: m.resultScore })}</Badge> : null}
                        </Stack>
                      ) : null}
                    </Stack>
                  ) : null}
                  <MarkdownBlocks content={m.content} />
                </Card>
              </Group>
            ))}
          </Stack>
        </ScrollArea>

        <Stack gap="xs" mt="md">
          <Group justify="space-between" align="center" wrap="wrap">
            <Group gap="xs" wrap="nowrap">
              <FileButton onChange={setImageFile} accept="image/*">
                {(props) => (
                  <Button
                    {...props}
                    variant={imageFile ? "light" : "subtle"}
                    leftSection={<IconPhoto size={16} />}
                  >
                    {imageFile ? t("agent.imageSelected") : t("agent.selectImage")}
                  </Button>
                )}
              </FileButton>
              {imageFile ? (
                <ActionIcon
                  variant="subtle"
                  color="red"
                  onClick={() => setImageFile(null)}
                  aria-label={t("agent.removeImage")}
                >
                  <IconTrash size={16} />
                </ActionIcon>
              ) : null}
            </Group>
            <Button
              leftSection={<IconSend size={16} />}
              loading={loading}
              disabled={!canSend}
              onClick={send}
              fullWidth={isMobile}
            >
              {t("agent.send")}
            </Button>
          </Group>

          {selectedPreviewUrl && imageFile ? (
            <Card withBorder radius="md" p="xs">
              <Group align="flex-start" wrap="nowrap">
                <Image
                  src={selectedPreviewUrl}
                  alt={imageFile.name}
                  radius="md"
                  h={84}
                  w={84}
                  fit="cover"
                />
                <Stack gap={2} style={{ minWidth: 0 }}>
                  <Text size="sm" fw={500} lineClamp={1}>
                    {imageFile.name}
                  </Text>
                  <Text size="xs" c="dimmed">
                    {formatFileSize(imageFile.size)}
                  </Text>
                  <Text size="xs" c="dimmed">
                    {t("agent.thumbnailHint")}
                  </Text>
                </Stack>
              </Group>
            </Card>
          ) : null}

          <Textarea
            minRows={2}
            maxRows={6}
            value={input}
            onChange={(e) => setInput(e.currentTarget.value)}
            placeholder={t("agent.inputPlaceholder")}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                void send();
              }
            }}
          />
          <Text size="xs" c="dimmed">
            {t("agent.keyboardHint")}
          </Text>
          <Switch
            checked={updateMilvus}
            onChange={(e) => setUpdateMilvus(e.currentTarget.checked)}
            label={t("agent.updateMilvusLabel")}
            description={t("agent.updateMilvusDescription")}
          />
          {encodingPercent !== null ? (
            <Text size="xs" c="dimmed">
              {t("agent.encodingProgress", { percent: encodingPercent })}
            </Text>
          ) : null}
        </Stack>
      </Card>
    </Stack>
  );
}
