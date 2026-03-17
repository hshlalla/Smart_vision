import React from "react";
import { ScrollArea, Table, Text } from "@mantine/core";

type Props = {
  content: string;
};

type Block =
  | { type: "text"; content: string }
  | { type: "table"; header: string[]; rows: string[][] };

function isSeparatorRow(line: string) {
  const cells = line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
  return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function parseTableRow(line: string) {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

function parseBlocks(content: string): Block[] {
  const lines = String(content || "").split("\n");
  const blocks: Block[] = [];
  let currentText: string[] = [];
  let idx = 0;

  const flushText = () => {
    const text = currentText.join("\n").trim();
    if (text) blocks.push({ type: "text", content: text });
    currentText = [];
  };

  while (idx < lines.length) {
    const line = lines[idx];
    const next = lines[idx + 1];
    const looksLikeTable = line.includes("|") && typeof next === "string" && isSeparatorRow(next);
    if (!looksLikeTable) {
      currentText.push(line);
      idx += 1;
      continue;
    }

    flushText();
    const header = parseTableRow(line);
    idx += 2;
    const rows: string[][] = [];
    while (idx < lines.length) {
      const rowLine = lines[idx];
      if (!rowLine.includes("|") || !rowLine.trim()) break;
      rows.push(parseTableRow(rowLine));
      idx += 1;
    }
    blocks.push({ type: "table", header, rows });
  }

  flushText();
  return blocks;
}

export default function MarkdownBlocks({ content }: Props) {
  const blocks = parseBlocks(content);
  return (
    <>
      {blocks.map((block, index) => {
        if (block.type === "text") {
          return (
            <Text key={`text-${index}`} size="sm" style={{ whiteSpace: "pre-wrap" }}>
              {block.content}
            </Text>
          );
        }
        return (
          <ScrollArea key={`table-${index}`} type="auto" offsetScrollbars>
            <Table striped withTableBorder withColumnBorders highlightOnHover horizontalSpacing="sm" verticalSpacing="xs">
              <Table.Thead>
                <Table.Tr>
                  {block.header.map((cell, cellIndex) => (
                    <Table.Th key={`h-${cellIndex}`}>{cell}</Table.Th>
                  ))}
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {block.rows.map((row, rowIndex) => (
                  <Table.Tr key={`r-${rowIndex}`}>
                    {block.header.map((_, cellIndex) => (
                      <Table.Td key={`c-${rowIndex}-${cellIndex}`}>{row[cellIndex] || ""}</Table.Td>
                    ))}
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </ScrollArea>
        );
      })}
    </>
  );
}
