import React from "react";
import { Navigate, Route, Routes } from "react-router-dom";

import { RequireAuth } from "./state/auth";
import LoginPage from "./views/LoginPage";
import AppShellLayout from "./views/AppShellLayout";
import AgentChatPage from "./views/AgentChatPage";
import SearchPage from "./views/SearchPage";
import IndexPage from "./views/IndexPage";
import CatalogPage from "./views/CatalogPage";

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />

      <Route
        path="/app"
        element={
          <RequireAuth>
            <AppShellLayout />
          </RequireAuth>
        }
      >
        <Route index element={<Navigate to="/app/search" replace />} />
        <Route path="chat" element={<AgentChatPage />} />
        <Route path="search" element={<SearchPage />} />
        <Route path="index" element={<IndexPage />} />
        <Route path="catalog" element={<CatalogPage />} />
      </Route>

      <Route path="*" element={<Navigate to="/app/search" replace />} />
    </Routes>
  );
}
