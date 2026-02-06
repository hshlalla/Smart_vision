import React, { useMemo, useState } from "react";
import {
  ActionIcon,
  Avatar,
  Badge,
  Button,
  Card,
  FileButton,
  Group,
  ScrollArea,
  Stack,
  Text,
  Textarea,
  Title,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { IconPhoto, IconSend, IconTrash } from "@tabler/icons-react";

import { useAuth } from "../state/auth";
import { apiFetchJson, toBase64 } from "../utils/api";

type ChatMsg = {
  role: "user" | "assistant";
  content: string;
};

export default function AgentChatPage() {
  const auth = useAuth();
  const [messages, setMessages] = useState<ChatMsg[]>([
    {
      role: "assistant",
      content:
        "이미지를 올리고 “이 제품 뭐야?”처럼 물어보면, 검색 결과 기반으로 설명하고 필요하면 gparts 가격 후보도 찾아줄게.",
    },
  ]);
  const [input, setInput] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);

  const canSend = useMemo(() => Boolean(input.trim() || imageFile), [input, imageFile]);

  async function send() {
    if (!canSend) return;
    const text = input.trim() || "이 제품 뭐야?";
    setInput("");
    setLoading(true);
    setMessages((prev) => [...prev, { role: "user", content: text }]);

    try {
      const image_base64 = imageFile ? await toBase64(imageFile) : null;
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (auth.token) headers.Authorization = `Bearer ${auth.token}`;

      const res = await apiFetchJson<{ answer: string; debug?: any }>("/api/v1/agent/chat", {
        method: "POST",
        headers,
        body: JSON.stringify({
          message: text,
          image_base64,
          max_tool_results: 5,
        }),
      });

      const sources = Array.isArray((res as any).sources) ? (res as any).sources : [];
      const sourcesText =
        sources.length > 0
          ? `\n\nSources:\n${sources
              .slice(0, 5)
              .map((s: any) => `- ${String(s.title || "").trim()} ${String(s.url || "").trim()}`.trim())
              .join("\n")}`
          : "";

      setMessages((prev) => [...prev, { role: "assistant", content: (res.answer || "(empty)") + sourcesText }]);
      setImageFile(null);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      notifications.show({ color: "red", title: "에이전트 오류", message: msg });
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `오류가 발생했어: ${msg}` },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <Stack gap="md" style={{ height: "calc(100vh - 110px)" }}>
      <Group justify="space-between" align="baseline">
        <Title order={3}>Agent Bot</Title>
        <Badge variant="light">LangChain tool agent</Badge>
      </Group>

      <Card withBorder radius="lg" p="md" style={{ flex: 1, display: "flex", flexDirection: "column" }}>
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
                  <Text size="sm" style={{ whiteSpace: "pre-wrap" }}>
                    {m.content}
                  </Text>
                </Card>
              </Group>
            ))}
          </Stack>
        </ScrollArea>

        <Stack gap="xs" mt="md">
          <Group justify="space-between">
            <Group gap="xs">
              <FileButton onChange={setImageFile} accept="image/*">
                {(props) => (
                  <Button
                    {...props}
                    variant={imageFile ? "light" : "subtle"}
                    leftSection={<IconPhoto size={16} />}
                  >
                    {imageFile ? "이미지 선택됨" : "이미지"}
                  </Button>
                )}
              </FileButton>
              {imageFile ? (
                <ActionIcon variant="subtle" color="red" onClick={() => setImageFile(null)} aria-label="remove image">
                  <IconTrash size={16} />
                </ActionIcon>
              ) : null}
            </Group>
            <Button
              leftSection={<IconSend size={16} />}
              loading={loading}
              disabled={!canSend}
              onClick={send}
            >
              보내기
            </Button>
          </Group>

          <Textarea
            minRows={2}
            maxRows={6}
            value={input}
            onChange={(e) => setInput(e.currentTarget.value)}
            placeholder='예: "이 제품 뭐야? 모델/제조사 추정해줘. 가격도 알려줘"'
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                void send();
              }
            }}
          />
          <Text size="xs" c="dimmed">
            Enter: 줄바꿈 · Ctrl/⌘+Enter: 전송
          </Text>
        </Stack>
      </Card>
    </Stack>
  );
}
