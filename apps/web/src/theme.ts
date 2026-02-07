import { createTheme, MantineColorsTuple } from "@mantine/core";

const brand: MantineColorsTuple = [
  "#edf2ff",
  "#dbe4ff",
  "#bac8ff",
  "#91a7ff",
  "#748ffc",
  "#5c7cfa",
  "#4c6ef5",
  "#4263eb",
  "#3b5bdb",
  "#364fc7",
];

export const theme = createTheme({
  primaryColor: "brand",
  colors: { brand },
  defaultRadius: "md",
  fontFamily:
    "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Noto Sans KR, Apple SD Gothic Neo, sans-serif",
});
