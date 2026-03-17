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

import LanguageToggle from "../components/LanguageToggle";
import { useAuth, useLoginFlow } from "../state/auth";
import { useI18n } from "../state/i18n";

export default function LoginPage() {
  const auth = useAuth();
  const { login } = useLoginFlow();
  const { t } = useI18n();
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
          <Group justify="space-between" align="flex-start">
            <Stack gap={6}>
              <Title order={2} c="white">
                {t("common.appName")}
              </Title>
              <Text c="dimmed">{t("login.tagline")}</Text>
            </Stack>
            <LanguageToggle />
          </Group>

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
                  label={t("login.username")}
                  leftSection={<IconUser size={16} />}
                  placeholder="admin"
                  value={username}
                  onChange={(e) => setUsername(e.currentTarget.value)}
                  required
                  disabled={auth.authEnabled === false}
                />
                <PasswordInput
                  label={t("login.password")}
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
                    {t("login.continueNoAuth")}
                  </Button>
                ) : (
                  <Button type="submit" loading={loading} fullWidth radius="md">
                    {t("login.signIn")}
                  </Button>
                )}
                <Group justify="space-between">
                  <Text size="sm" c="dimmed">
                    {auth.authEnabled === false
                      ? t("login.authDisabled")
                      : t("login.authRequired")}
                  </Text>
                  <Anchor
                    size="sm"
                    href={`${import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"}/api/docs`}
                    target="_blank"
                    c="dimmed"
                  >
                    {t("common.apiDocs")}
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
