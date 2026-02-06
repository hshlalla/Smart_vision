import React, { createContext, useContext, useEffect, useMemo, useState } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";
import { notifications } from "@mantine/notifications";

import { apiFetchJson } from "../utils/api";

type AuthUser = {
  username: string;
};

type AuthContextValue = {
  token: string | null;
  user: AuthUser | null;
  authEnabled: boolean | null;
  authReady: boolean;
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  refreshMe: () => Promise<void>;
};

const TOKEN_KEY = "sv_auth_token";

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [token, setToken] = useState<string | null>(() => localStorage.getItem(TOKEN_KEY));
  const [user, setUser] = useState<AuthUser | null>(null);
  const [authEnabled, setAuthEnabled] = useState<boolean | null>(null);

  useEffect(() => {
    apiFetchJson<{ enabled: boolean }>("/api/v1/auth/status")
      .then((res) => setAuthEnabled(Boolean(res.enabled)))
      .catch(() => setAuthEnabled(true)); // safest default: require auth
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      token,
      user,
      authEnabled,
      authReady: authEnabled !== null,
      isAuthenticated: authEnabled === false ? true : Boolean(token),
      async login(username, password) {
        const res = await apiFetchJson<{ access_token: string; token_type: string; username: string }>(
          "/api/v1/auth/login",
          {
            method: "POST",
            body: JSON.stringify({ username, password }),
            headers: { "Content-Type": "application/json" },
          },
        );

        localStorage.setItem(TOKEN_KEY, res.access_token);
        setToken(res.access_token);
        setUser({ username: res.username });
      },
      logout() {
        localStorage.removeItem(TOKEN_KEY);
        setToken(null);
        setUser(null);
      },
      async refreshMe() {
        if (!token) return;
        const me = await apiFetchJson<{ username: string }>("/api/v1/auth/me", {
          headers: { Authorization: `Bearer ${token}` },
        });
        setUser({ username: me.username });
      },
    }),
    [token, user, authEnabled],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("AuthContext not initialized");
  return ctx;
}

export function RequireAuth({ children }: { children: React.ReactNode }) {
  const auth = useAuth();
  const location = useLocation();
  if (!auth.authReady) {
    return null;
  }
  if (!auth.isAuthenticated) {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  }
  return children;
}

export function useLoginFlow() {
  const navigate = useNavigate();
  const location = useLocation() as { state?: { from?: string } };
  const auth = useAuth();

  return {
    async login(username: string, password: string) {
      try {
        await auth.login(username, password);
        const dest = location.state?.from || "/app/search";
        navigate(dest, { replace: true });
        notifications.show({ color: "teal", title: "로그인 성공", message: "환영합니다." });
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        notifications.show({ color: "red", title: "로그인 실패", message: msg });
        throw err;
      }
    },
  };
}
