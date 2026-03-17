import React from "react";
import { SegmentedControl } from "@mantine/core";

import { useI18n } from "../state/i18n";

export default function LanguageToggle() {
  const { language, setLanguage } = useI18n();

  return (
    <SegmentedControl
      size="xs"
      radius="xl"
      value={language}
      onChange={(value) => setLanguage(value as "ko" | "en")}
      data={[
        { label: "KO", value: "ko" },
        { label: "EN", value: "en" },
      ]}
    />
  );
}
