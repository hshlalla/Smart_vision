import React, { useMemo, useState } from "react";
import {
  Badge,
  Button,
  Card,
  FileInput,
  Grid,
  Group,
  NumberInput,
  Stack,
  Text,
  TextInput,
  Title,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { IconPhoto, IconSearch } from "@tabler/icons-react";

import { useAuth } from "../state/auth";
import { apiFetchJson, toBase64 } from "../utils/api";

type SearchResult = Record<string, any>;

export default function SearchPage() {
  const auth = useAuth();
  const [queryText, setQueryText] = useState("");
  const [partNumber, setPartNumber] = useState("");
  const [topK, setTopK] = useState<number>(10);
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);

  const hasQuery = useMemo(() => Boolean(queryText.trim() || file), [queryText, file]);

  async function runSearch() {
    if (!hasQuery) {
      notifications.show({ color: "yellow", title: "입력 필요", message: "텍스트 또는 이미지를 넣어주세요." });
      return;
    }
    setLoading(true);
    try {
      const image_base64 = file ? await toBase64(file) : null;
      const payload = {
        query_text: queryText.trim() || null,
        image_base64,
        part_number: partNumber.trim() || null,
        top_k: topK,
      };
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (auth.token) headers.Authorization = `Bearer ${auth.token}`;
      const res = await apiFetchJson<{ results: SearchResult[] }>("/api/v1/hybrid/search", {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
      });
      setResults(res.results || []);
      notifications.show({ color: "teal", title: "검색 완료", message: `결과 ${res.results?.length || 0}건` });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      notifications.show({ color: "red", title: "검색 실패", message: msg });
    } finally {
      setLoading(false);
    }
  }

  return (
    <Stack gap="md">
      <Group justify="space-between" align="baseline">
        <Title order={3}>Hybrid Search</Title>
        <Badge variant="light">image + OCR + caption + metadata</Badge>
      </Group>

      <Card withBorder radius="lg" p="lg">
        <Grid gutter="md">
          <Grid.Col span={{ base: 12, md: 7 }}>
            <TextInput
              label="텍스트 쿼리"
              placeholder="예: 520d 헤드램프, PN-1234, maker..."
              value={queryText}
              onChange={(e) => setQueryText(e.currentTarget.value)}
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 5 }}>
            <TextInput
              label="Part number 필터(옵션)"
              placeholder="예: PN-0000"
              value={partNumber}
              onChange={(e) => setPartNumber(e.currentTarget.value)}
            />
          </Grid.Col>

          <Grid.Col span={{ base: 12, md: 7 }}>
            <FileInput
              label="이미지(옵션)"
              placeholder="사진을 선택하세요"
              leftSection={<IconPhoto size={16} />}
              value={file}
              onChange={setFile}
              accept="image/*"
              clearable
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 5 }}>
            <NumberInput
              label="Top-K"
              value={topK}
              onChange={(v) => setTopK(typeof v === "number" ? v : 10)}
              min={1}
              max={50}
            />
          </Grid.Col>

          <Grid.Col span={12}>
            <Group justify="flex-end">
              <Button leftSection={<IconSearch size={16} />} loading={loading} onClick={runSearch}>
                검색
              </Button>
            </Group>
          </Grid.Col>
        </Grid>
      </Card>

      <Stack gap="sm">
        {results.length === 0 ? (
          <Card withBorder radius="lg" p="lg">
            <Text c="dimmed">검색 결과가 여기에 표시됩니다.</Text>
          </Card>
        ) : (
          results.map((r, idx) => (
            <Card key={idx} withBorder radius="lg" p="lg">
              <Group justify="space-between" align="flex-start">
                <Stack gap={2}>
                  <Text fw={700}>{String(r.model_id || "unknown")}</Text>
                  <Text size="sm" c="dimmed">
                    {String(r.description || "")}
                  </Text>
                </Stack>
                <Badge variant="light">
                  score {typeof r.score === "number" ? r.score.toFixed(3) : String(r.score ?? "-")}
                </Badge>
              </Group>
              <Group gap="xs" mt="sm">
                {r.maker ? <Badge variant="outline">maker: {String(r.maker)}</Badge> : null}
                {r.part_number ? <Badge variant="outline">pn: {String(r.part_number)}</Badge> : null}
                {r.category ? <Badge variant="outline">cat: {String(r.category)}</Badge> : null}
                {r.lexical_hit ? <Badge color="grape" variant="light">lexical</Badge> : null}
              </Group>
              {r.ocr_text ? (
                <Text size="sm" c="dimmed" mt="sm" lineClamp={3}>
                  OCR: {String(r.ocr_text)}
                </Text>
              ) : null}
              {r.caption_text ? (
                <Text size="sm" c="dimmed" mt={6} lineClamp={2}>
                  Caption: {String(r.caption_text)}
                </Text>
              ) : null}
            </Card>
          ))
        )}
      </Stack>
    </Stack>
  );
}
