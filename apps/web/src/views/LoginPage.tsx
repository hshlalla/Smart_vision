import React, { useState } from "react";
import {
  Anchor,
  Box,
  Button,
  Card,
  Container,
  Group,
  PasswordInput,
  Stack,
  Text,
  TextInput,
  Title,
} from "@mantine/core";
import { IconLock, IconUser } from "@tabler/icons-react";

import { useAuth, useLoginFlow } from "../state/auth";

export default function LoginPage() {
  const auth = useAuth();
  const { login } = useLoginFlow();
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    try {
      await login(username, password);
    } finally {
      setLoading(false);
    }
  }

  return (
    <Box
      style={{
        minHeight: "100vh",
        background:
          "radial-gradient(1200px 800px at 20% 10%, rgba(76,110,245,0.30), transparent 60%), radial-gradient(900px 700px at 80% 30%, rgba(94,234,212,0.18), transparent 55%), linear-gradient(180deg, rgba(13,17,23,1), rgba(10,12,16,1))",
      }}
    >
      <Container size="xs" py={80}>
        <Stack gap="lg">
          <Stack gap={6}>
            <Title order={2} c="white">
              Smart Vision
            </Title>
            <Text c="dimmed">
              모바일에서도 접속 가능한 검색/인덱싱 UI (Gradio 대체)
            </Text>
          </Stack>

          <Card
            withBorder
            radius="lg"
            p="xl"
            style={{
              backdropFilter: "blur(10px)",
              backgroundColor: "rgba(255,255,255,0.06)",
              borderColor: "rgba(255,255,255,0.12)",
            }}
          >
            <form onSubmit={onSubmit}>
              <Stack gap="md">
                <TextInput
                  label="아이디"
                  leftSection={<IconUser size={16} />}
                  placeholder="admin"
                  value={username}
                  onChange={(e) => setUsername(e.currentTarget.value)}
                  required
                  disabled={auth.authEnabled === false}
                />
                <PasswordInput
                  label="비밀번호"
                  leftSection={<IconLock size={16} />}
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.currentTarget.value)}
                  required
                  disabled={auth.authEnabled === false}
                />
                {auth.authEnabled === false ? (
                  <Button
                    fullWidth
                    radius="md"
                    variant="light"
                    onClick={() => (window.location.href = "/app/search")}
                  >
                    인증이 꺼져있음 — 계속하기
                  </Button>
                ) : (
                  <Button type="submit" loading={loading} fullWidth radius="md">
                    로그인
                  </Button>
                )}
                <Group justify="space-between">
                  <Text size="sm" c="dimmed">
                    {auth.authEnabled === false
                      ? "API 인증이 비활성화되어 있습니다."
                      : "API에서 인증을 켜면 로그인 필수"}
                  </Text>
                  <Anchor
                    size="sm"
                    href={`${import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"}/api/docs`}
                    target="_blank"
                    c="dimmed"
                  >
                    API Docs
                  </Anchor>
                </Group>
              </Stack>
            </form>
          </Card>
        </Stack>
      </Container>
    </Box>
  );
}
