import React, { useEffect } from "react";
import {
  AppShell,
  Burger,
  Group,
  NavLink,
  ScrollArea,
  Text,
  Title,
  UnstyledButton,
  Avatar,
  Menu,
  rem,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { IconDatabaseSearch, IconLogout, IconMessageChatbot, IconSearch, IconUpload } from "@tabler/icons-react";
import { Outlet, useLocation, useNavigate } from "react-router-dom";

import { useAuth } from "../state/auth";
import LanguageToggle from "../components/LanguageToggle";
import { useI18n } from "../state/i18n";

export default function AppShellLayout() {
  const [opened, { toggle, close }] = useDisclosure();
  const navigate = useNavigate();
  const location = useLocation();
  const auth = useAuth();
  const { t } = useI18n();

  useEffect(() => {
    close();
  }, [location.pathname, close]);

  useEffect(() => {
    auth.refreshMe().catch(() => {
      // token might be stale; keep UX simple: let the user re-login
      auth.logout();
      navigate("/login", { replace: true });
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const navItems = [
    { label: t("shell.navChat"), to: "/app/chat", icon: <IconMessageChatbot size={18} /> },
    { label: t("shell.navSearch"), to: "/app/search", icon: <IconSearch size={18} /> },
    { label: t("shell.navIndex"), to: "/app/index", icon: <IconUpload size={18} /> },
    { label: t("shell.navCatalog"), to: "/app/catalog", icon: <IconDatabaseSearch size={18} /> },
  ];

  return (
    <AppShell
      header={{ height: 64 }}
      navbar={{
        width: 280,
        breakpoint: "sm",
        collapsed: { mobile: !opened },
      }}
      padding="md"
    >
      <AppShell.Header
        style={{
          background:
            "linear-gradient(90deg, rgba(76,110,245,0.22), rgba(94,234,212,0.10))",
          borderBottomColor: "rgba(255,255,255,0.10)",
        }}
      >
        <Group h="100%" px="md" justify="space-between">
          <Group gap="sm">
            <Burger opened={opened} onClick={toggle} hiddenFrom="sm" size="sm" />
            <Title order={4}>{t("common.appName")}</Title>
            <Text size="sm" c="dimmed" visibleFrom="md">
              {t("shell.subtitle")}
            </Text>
          </Group>

          <Group gap="xs">
            <LanguageToggle />
            <Menu position="bottom-end" withinPortal>
            <Menu.Target>
              <UnstyledButton>
                <Group gap="sm">
                  <Avatar radius="xl" size="sm" />
                  <Text size="sm" fw={600} visibleFrom="sm">
                    {auth.user?.username || t("common.user")}
                  </Text>
                </Group>
              </UnstyledButton>
            </Menu.Target>
            <Menu.Dropdown>
              <Menu.Label>{t("common.account")}</Menu.Label>
              <Menu.Item
                leftSection={<IconLogout style={{ width: rem(16), height: rem(16) }} />}
                onClick={() => {
                  auth.logout();
                  navigate("/login", { replace: true });
                }}
              >
                {t("common.logout")}
              </Menu.Item>
            </Menu.Dropdown>
            </Menu>
          </Group>
        </Group>
      </AppShell.Header>

      <AppShell.Navbar p="xs">
        <AppShell.Section component={ScrollArea} grow>
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              label={item.label}
              leftSection={item.icon}
              active={location.pathname === item.to}
              onClick={() => navigate(item.to)}
              variant="subtle"
              style={{ borderRadius: rem(10) }}
            />
          ))}
        </AppShell.Section>

        <AppShell.Section p="xs">
          <Text size="xs" c="dimmed">
            {t("common.api")}: {((import.meta as any).env?.VITE_API_BASE_URL || "(proxy:/api)").toString()}
          </Text>
        </AppShell.Section>
      </AppShell.Navbar>

      <AppShell.Main>
        <Outlet />
      </AppShell.Main>
    </AppShell>
  );
}
