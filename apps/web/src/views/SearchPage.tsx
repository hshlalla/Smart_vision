import React, { useEffect, useMemo, useState } from "react";
import {
  AspectRatio,
  Badge,
  Button,
  Card,
  FileInput,
  Grid,
  Group,
  NumberInput,
  Switch,
  Stack,
  Text,
  TextInput,
  Textarea,
  Title,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { IconPhoto, IconSearch, IconSparkles, IconUpload } from "@tabler/icons-react";

import { useAuth } from "../state/auth";
import { useI18n } from "../state/i18n";
import { apiFetchJson, toBase64 } from "../utils/api";

type SearchResult = Record<string, any>;
type MetadataDraft = {
  model_id: string;
  maker: string;
  part_number: string;
  category: string;
  description: string;
  product_info: string;
  price_value: number | null;
  source?: string;
};

type IndexTaskStatus = {
  task_id: string;
  status: string;
  model_id?: string | null;
  detail: string;
};

type PreviewResponse = {
  draft: MetadataDraft;
  ocr_image_indices: number[];
};

const TASK_TERMINAL_STATES = new Set(["completed", "failed"]);

const EMPTY_DRAFT: MetadataDraft = {
  model_id: "",
  maker: "",
  part_number: "",
  category: "",
  description: "",
  product_info: "",
  price_value: null,
  source: "",
};

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

export default function SearchPage() {
  const auth = useAuth();
  const { t, statusLabel } = useI18n();
  const [queryText, setQueryText] = useState("");
  const [partNumber, setPartNumber] = useState("");
  const [topK, setTopK] = useState<number>(10);
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [encodingPercent, setEncodingPercent] = useState<number | null>(null);
  const [useReranker, setUseReranker] = useState(false);
  const [draft, setDraft] = useState<MetadataDraft>(EMPTY_DRAFT);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [confirmLoading, setConfirmLoading] = useState(false);
  const [taskStatus, setTaskStatus] = useState<IndexTaskStatus | null>(null);
  const [ocrImageIndices, setOcrImageIndices] = useState<number[]>([]);

  const hasQuery = useMemo(() => Boolean(queryText.trim() || file), [queryText, file]);

  async function runSearch() {
    if (!hasQuery) {
      notifications.show({ color: "yellow", title: t("search.inputNeededTitle"), message: t("search.inputNeededMessage") });
      return;
    }
    const selected = file;
    setLoading(true);
    try {
      if (selected) {
        console.info("[HybridSearch] preparing upload", {
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
      const payload = {
        query_text: queryText.trim() || null,
        image_base64,
        part_number: partNumber.trim() || null,
        top_k: topK,
        use_reranker: useReranker,
      };
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (auth.token) headers.Authorization = `Bearer ${auth.token}`;
      const res = await apiFetchJson<{ results: SearchResult[] }>("/api/v1/hybrid/search", {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
      });
      setResults(res.results || []);
      if ((res.results || []).length > 0) {
        setDraft(EMPTY_DRAFT);
      }
      console.info("[HybridSearch] request completed", {
        hasImage: Boolean(selected),
        resultCount: res.results?.length || 0,
      });
      notifications.show({
        color: "teal",
        title: t("search.completedTitle"),
        message: t("search.completedMessage", { count: res.results?.length || 0 }),
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error("[HybridSearch] request failed", { hasImage: Boolean(selected), error: msg });
      notifications.show({ color: "red", title: t("search.failedTitle"), message: msg });
    } finally {
      setLoading(false);
      setEncodingPercent(null);
    }
  }

  function setDraftField<K extends keyof MetadataDraft>(key: K, value: MetadataDraft[K]) {
    setDraft((prev) => ({ ...prev, [key]: value }));
  }

  async function encodeSelectedFile(): Promise<string> {
    if (!file) {
      throw new Error(t("search.imageNeededSaveMessage"));
    }
    return toBase64(file, {
      maxBytes: 5 * 1024 * 1024,
      maxDimension: 1600,
      quality: 0.82,
      onProgress: (pct) => setEncodingPercent(pct),
    });
  }

  async function runWritebackPreview() {
    if (!file) {
      notifications.show({ color: "yellow", title: t("search.imageNeededTitle"), message: t("search.imageNeededDraftMessage") });
      return;
    }
    setPreviewLoading(true);
    try {
      const image_base64 = await encodeSelectedFile();
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (auth.token) headers.Authorization = `Bearer ${auth.token}`;
      const res = await apiFetchJson<PreviewResponse>("/api/v1/hybrid/index/preview", {
        method: "POST",
        headers,
        body: JSON.stringify({ image_base64 }),
      });
      setDraft({
        model_id: res.draft.model_id || "",
        maker: res.draft.maker || "",
        part_number: res.draft.part_number || "",
        category: res.draft.category || "",
        description: res.draft.description || "",
        product_info: res.draft.product_info || "",
        price_value: typeof res.draft.price_value === "number" ? res.draft.price_value : null,
        source: res.draft.source || "openai",
      });
      setOcrImageIndices(Array.isArray(res.ocr_image_indices) ? res.ocr_image_indices : []);
      notifications.show({
        color: "teal",
        title: t("search.draftCompletedTitle"),
        message: t("search.draftCompletedMessage"),
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setOcrImageIndices([]);
      notifications.show({ color: "red", title: t("search.draftFailedTitle"), message: msg });
    } finally {
      setPreviewLoading(false);
      setEncodingPercent(null);
    }
  }

  async function confirmWriteback() {
    if (!file) {
      notifications.show({ color: "yellow", title: t("search.imageNeededTitle"), message: t("search.imageNeededSaveMessage") });
      return;
    }
    setConfirmLoading(true);
    try {
      const image_base64 = await encodeSelectedFile();
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (auth.token) headers.Authorization = `Bearer ${auth.token}`;
      const res = await apiFetchJson<{ status: string; model_id?: string; task_id?: string }>("/api/v1/hybrid/index/confirm", {
        method: "POST",
        headers,
        body: JSON.stringify({ image_base64, ocr_image_indices: ocrImageIndices, ...draft }),
      });
      setTaskStatus(
        res.task_id
          ? {
              task_id: res.task_id,
              status: res.status,
              model_id: res.model_id ?? null,
              detail: "백그라운드 인덱싱 작업이 시작되었습니다.",
            }
          : null,
      );
      notifications.show({
        color: "blue",
        title: t("search.saveStartedTitle"),
        message: res.model_id
          ? t("search.saveStartedWithModel", { modelId: res.model_id })
          : t("search.saveStartedWithoutModel"),
      });
      if (res.model_id) {
        setDraft((prev) => ({ ...prev, model_id: res.model_id || prev.model_id }));
      }
      if (res.task_id) {
        setOcrImageIndices([]);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      notifications.show({ color: "red", title: t("search.saveFailedTitle"), message: msg });
    } finally {
      setConfirmLoading(false);
      setEncodingPercent(null);
    }
  }

  useEffect(() => {
    if (!taskStatus?.task_id) return;
    if (TASK_TERMINAL_STATES.has(taskStatus.status)) return;

    let cancelled = false;
    let timer: number | null = null;
    const headers: Record<string, string> = {};
    if (auth.token) headers.Authorization = `Bearer ${auth.token}`;

    const poll = async () => {
      try {
        const res = await apiFetchJson<IndexTaskStatus>(`/api/v1/hybrid/index/tasks/${taskStatus.task_id}`, {
          headers,
        });
        if (cancelled) return;
        setTaskStatus(res);
        if (res.model_id) {
          setDraft((prev) => ({ ...prev, model_id: res.model_id || prev.model_id }));
        }
        if (res.status === "completed") {
          notifications.show({
            color: "teal",
            title: t("search.saveCompletedTitle"),
            message: res.model_id ? `${res.detail} (${res.model_id})` : res.detail,
          });
          return;
        }
        if (res.status === "failed") {
          notifications.show({
            color: "red",
            title: t("search.saveFailedTitle"),
            message: res.detail || t("search.saveFailedFallback"),
          });
          return;
        }
        timer = window.setTimeout(poll, 2500);
      } catch (err) {
        if (cancelled) return;
        const msg = err instanceof Error ? err.message : String(err);
        setTaskStatus((prev) =>
          prev
            ? {
                ...prev,
                detail: t("search.retryingStatus", { message: msg }),
              }
            : prev,
        );
        timer = window.setTimeout(poll, 4000);
      }
    };

    timer = window.setTimeout(poll, 1500);
    return () => {
      cancelled = true;
      if (timer !== null) {
        window.clearTimeout(timer);
      }
    };
  }, [auth.token, taskStatus?.task_id, taskStatus?.status]);

  return (
    <Stack gap="md">
      <Group justify="space-between" align="baseline">
        <Title order={3}>{t("search.title")}</Title>
        <Badge variant="light">{t("search.badge")}</Badge>
      </Group>

      <Card withBorder radius="lg" p="lg">
        <Grid gutter="md">
          <Grid.Col span={{ base: 12, md: 7 }}>
            <TextInput
              label={t("search.queryLabel")}
              placeholder={t("search.queryPlaceholder")}
              value={queryText}
              onChange={(e) => setQueryText(e.currentTarget.value)}
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 5 }}>
            <TextInput
              label={t("search.partFilterLabel")}
              placeholder={t("search.partFilterPlaceholder")}
              value={partNumber}
              onChange={(e) => setPartNumber(e.currentTarget.value)}
            />
          </Grid.Col>

          <Grid.Col span={{ base: 12, md: 7 }}>
            <FileInput
              label={t("search.imageLabel")}
              placeholder={t("search.imagePlaceholder")}
              leftSection={<IconPhoto size={16} />}
              value={file}
              onChange={setFile}
              accept="image/*"
              clearable
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 5 }}>
            <NumberInput
              label={t("search.topKLabel")}
              value={topK}
              onChange={(v) => setTopK(typeof v === "number" ? v : 10)}
              min={1}
              max={50}
            />
          </Grid.Col>

          <Grid.Col span={12}>
            <Group justify="space-between" align="center">
              <Switch
                label={t("search.rerankerLabel")}
                checked={useReranker}
                onChange={(event) => setUseReranker(event.currentTarget.checked)}
              />
              <Text size="xs" c="dimmed">
                {t("search.rerankerDescription")}
              </Text>
            </Group>
          </Grid.Col>

          <Grid.Col span={12}>
            <Group justify="flex-end">
              <Button leftSection={<IconSearch size={16} />} loading={loading} onClick={runSearch}>
                {t("search.searchButton")}
              </Button>
            </Group>
            {encodingPercent !== null ? (
              <Text size="xs" c="dimmed" mt={6}>
                {t("search.encodingProgress", { percent: encodingPercent })}
              </Text>
            ) : null}
          </Grid.Col>
        </Grid>
      </Card>

      <Stack gap="sm">
        {results.length === 0 ? (
          <Card withBorder radius="lg" p="lg">
            <Stack gap="sm">
              <Text c="dimmed">{t("search.noResults")}</Text>
              {file ? (
                <>
                  <Group justify="space-between">
                    <Text fw={600}>{t("search.writebackTitle")}</Text>
                    <Button
                      size="sm"
                      variant="light"
                      leftSection={<IconSparkles size={16} />}
                      loading={previewLoading}
                      onClick={runWritebackPreview}
                    >
                      {t("search.createDraftButton")}
                    </Button>
                  </Group>
                  {(draft.description || draft.maker || draft.part_number || draft.category || draft.product_info) ? (
                    <Grid gutter="md">
                      <Grid.Col span={{ base: 12, md: 6 }}>
                        <TextInput
                          label={t("search.modelIdLabel")}
                          placeholder={t("search.modelIdPlaceholder")}
                          value={draft.model_id}
                          onChange={(e) => setDraftField("model_id", e.currentTarget.value)}
                        />
                      </Grid.Col>
                      <Grid.Col span={{ base: 12, md: 6 }}>
                        <TextInput
                          label={t("search.makerLabel")}
                          value={draft.maker}
                          onChange={(e) => setDraftField("maker", e.currentTarget.value)}
                        />
                      </Grid.Col>
                      <Grid.Col span={{ base: 12, md: 6 }}>
                        <TextInput
                          label={t("search.partNumberLabel")}
                          value={draft.part_number}
                          onChange={(e) => setDraftField("part_number", e.currentTarget.value)}
                        />
                      </Grid.Col>
                      <Grid.Col span={{ base: 12, md: 6 }}>
                        <TextInput
                          label={t("search.categoryLabel")}
                          value={draft.category}
                          onChange={(e) => setDraftField("category", e.currentTarget.value)}
                        />
                      </Grid.Col>
                      <Grid.Col span={{ base: 12, md: 6 }}>
                        <TextInput
                          label={t("search.productInfoLabel")}
                          value={draft.product_info}
                          onChange={(e) => setDraftField("product_info", e.currentTarget.value)}
                        />
                      </Grid.Col>
                      <Grid.Col span={{ base: 12, md: 6 }}>
                        <NumberInput
                          label={t("search.priceValueLabel")}
                          value={draft.price_value ?? undefined}
                          onChange={(v) => setDraftField("price_value", typeof v === "number" ? v : null)}
                          min={0}
                          thousandSeparator=","
                        />
                      </Grid.Col>
                      <Grid.Col span={12}>
                        <Textarea
                          label={t("search.descriptionLabel")}
                          minRows={3}
                          value={draft.description}
                          onChange={(e) => setDraftField("description", e.currentTarget.value)}
                        />
                      </Grid.Col>
                      <Grid.Col span={12}>
                        <Group justify="space-between">
                          <Group gap="xs">
                            {draft.source ? <Badge variant="light">{draft.source}</Badge> : null}
                            {taskStatus ? <Badge variant="light">{t("search.jobBadge", { status: statusLabel(taskStatus.status) })}</Badge> : null}
                          </Group>
                          <Button
                            size="sm"
                            leftSection={<IconUpload size={16} />}
                            loading={confirmLoading}
                            onClick={confirmWriteback}
                          >
                            {t("search.saveButton")}
                          </Button>
                        </Group>
                        {taskStatus?.detail ? (
                          <Text size="xs" c="dimmed" mt={6}>
                            {taskStatus.detail}
                          </Text>
                        ) : null}
                      </Grid.Col>
                    </Grid>
                  ) : null}
                </>
              ) : null}
            </Stack>
          </Card>
        ) : (
          results.map((r, idx) => (
            <Card key={idx} withBorder radius="lg" p="lg">
              <Grid gutter="md">
                <Grid.Col span={{ base: 12, sm: 4 }}>
                  {(() => {
                    const firstImage = Array.isArray(r.images) ? r.images[0] : null;
                    const mediaUrl = resolveMediaUrl(firstImage?.image_path);
                    if (!mediaUrl) {
                      return (
                        <AspectRatio ratio={4 / 3}>
                          <Card withBorder radius="md" p="md">
                            <Stack h="100%" justify="center" align="center" gap={4}>
                              <IconPhoto size={24} />
                              <Text size="xs" c="dimmed">
                                {t("search.imageEmpty")}
                              </Text>
                            </Stack>
                          </Card>
                        </AspectRatio>
                      );
                    }
                    return (
                      <AspectRatio ratio={4 / 3}>
                        <img
                          src={mediaUrl}
                          alt={String(r.model_id || "search-result")}
                          style={{ width: "100%", height: "100%", objectFit: "cover", borderRadius: 12 }}
                        />
                      </AspectRatio>
                    );
                  })()}
                </Grid.Col>
                <Grid.Col span={{ base: 12, sm: 8 }}>
                  <Group justify="space-between" align="flex-start">
                    <Stack gap={2}>
                      <Text fw={700}>{String(r.model_id || t("search.unknownModel"))}</Text>
                      <Text size="sm" c="dimmed">{String(r.description || "")}</Text>
                    </Stack>
                    <Badge variant="light">
                      {t("search.scoreBadge", {
                        score: typeof r.score === "number" ? r.score.toFixed(3) : String(r.score ?? "-"),
                      })}
                    </Badge>
                  </Group>
                  <Group gap="xs" mt="sm">
                    {r.maker ? <Badge variant="outline">{t("search.makerBadge", { value: String(r.maker) })}</Badge> : null}
                    {r.part_number ? <Badge variant="outline">{t("search.pnBadge", { value: String(r.part_number) })}</Badge> : null}
                    {r.category ? <Badge variant="outline">{t("search.categoryBadge", { value: String(r.category) })}</Badge> : null}
                    {r.lexical_hit ? <Badge color="grape" variant="light">{t("search.lexicalBadge")}</Badge> : null}
                  </Group>
                  {r.ocr_text ? (
                    <Text size="sm" c="dimmed" mt="sm" lineClamp={3}>
                      {t("search.ocrPrefix", { value: String(r.ocr_text) })}
                    </Text>
                  ) : null}
                  {r.caption_text ? (
                    <Text size="sm" c="dimmed" mt={6} lineClamp={2}>
                      {t("search.captionPrefix", { value: String(r.caption_text) })}
                    </Text>
                  ) : null}
                </Grid.Col>
              </Grid>
            </Card>
          ))
        )}
      </Stack>
    </Stack>
  );
}
