import React, { useState } from "react";
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
import { IconDatabaseSearch, IconFileSearch, IconFileTypePdf, IconUpload } from "@tabler/icons-react";

import { useAuth } from "../state/auth";
import { apiFetchJson } from "../utils/api";

type CatalogSearchResult = {
  score: number;
  document_id: string;
  source: string;
  page: number;
  chunk_id: number;
  model_id: string;
  part_number: string;
  maker: string;
  text: string;
};

export default function CatalogPage() {
  const auth = useAuth();

  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [source, setSource] = useState("");
  const [indexModelId, setIndexModelId] = useState("");
  const [indexPartNumber, setIndexPartNumber] = useState("");
  const [indexMaker, setIndexMaker] = useState("");
  const [indexLoading, setIndexLoading] = useState(false);

  const [queryText, setQueryText] = useState("");
  const [searchModelId, setSearchModelId] = useState("");
  const [searchPartNumber, setSearchPartNumber] = useState("");
  const [topK, setTopK] = useState<number>(10);
  const [searchLoading, setSearchLoading] = useState(false);
  const [results, setResults] = useState<CatalogSearchResult[]>([]);

  async function indexPdf() {
    if (!pdfFile) {
      notifications.show({ color: "yellow", title: "PDF 필요", message: "인덱싱할 PDF를 선택하세요." });
      return;
    }

    setIndexLoading(true);
    try {
      const form = new FormData();
      form.append("pdf", pdfFile);
      form.append("source", source.trim());
      form.append("model_id", indexModelId.trim());
      form.append("part_number", indexPartNumber.trim());
      form.append("maker", indexMaker.trim());

      const headers: Record<string, string> = {};
      if (auth.token) headers.Authorization = `Bearer ${auth.token}`;

      const res = await apiFetchJson<{
        status: string;
        source: string;
        pages_indexed: number;
        chunks_indexed: number;
      }>("/api/v1/catalog/index_pdf", {
        method: "POST",
        headers,
        body: form,
      });

      notifications.show({
        color: "teal",
        title: "카탈로그 인덱싱 완료",
        message: `${res.source} · ${res.pages_indexed}페이지 · ${res.chunks_indexed}청크`,
      });
      setPdfFile(null);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      notifications.show({ color: "red", title: "인덱싱 실패", message: msg });
    } finally {
      setIndexLoading(false);
    }
  }

  async function searchCatalog() {
    if (!queryText.trim()) {
      notifications.show({ color: "yellow", title: "질문 필요", message: "검색할 텍스트를 입력하세요." });
      return;
    }

    setSearchLoading(true);
    try {
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (auth.token) headers.Authorization = `Bearer ${auth.token}`;

      const res = await apiFetchJson<{ results: CatalogSearchResult[] }>("/api/v1/catalog/search", {
        method: "POST",
        headers,
        body: JSON.stringify({
          query_text: queryText.trim(),
          top_k: topK,
          model_id: searchModelId.trim() || null,
          part_number: searchPartNumber.trim() || null,
        }),
      });
      setResults(res.results || []);
      notifications.show({ color: "teal", title: "검색 완료", message: `결과 ${res.results?.length || 0}건` });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      notifications.show({ color: "red", title: "검색 실패", message: msg });
    } finally {
      setSearchLoading(false);
    }
  }

  return (
    <Stack gap="md">
      <Group justify="space-between" align="baseline">
        <Title order={3}>Catalog RAG</Title>
        <Badge variant="light">PDF index + semantic retrieval</Badge>
      </Group>

      <Card withBorder radius="lg" p="lg">
        <Group justify="space-between" mb="sm">
          <Text fw={700}>PDF 카탈로그 인덱싱</Text>
          <IconFileTypePdf size={18} />
        </Group>

        <Grid gutter="md">
          <Grid.Col span={12}>
            <FileInput
              label="PDF 파일"
              placeholder="카탈로그 PDF를 선택하세요"
              accept="application/pdf"
              value={pdfFile}
              onChange={setPdfFile}
              leftSection={<IconFileTypePdf size={16} />}
              clearable
              required
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}>
            <TextInput
              label="Source 이름(옵션)"
              placeholder="예: Internal Catalog 2025"
              value={source}
              onChange={(e) => setSource(e.currentTarget.value)}
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}>
            <TextInput
              label="Maker(옵션)"
              placeholder="예: ACME"
              value={indexMaker}
              onChange={(e) => setIndexMaker(e.currentTarget.value)}
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}>
            <TextInput
              label="Model ID(옵션)"
              placeholder="예: NB-ADP-16V"
              value={indexModelId}
              onChange={(e) => setIndexModelId(e.currentTarget.value)}
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}>
            <TextInput
              label="Part Number(옵션)"
              placeholder="예: ADP-16V-3A"
              value={indexPartNumber}
              onChange={(e) => setIndexPartNumber(e.currentTarget.value)}
            />
          </Grid.Col>
          <Grid.Col span={12}>
            <Group justify="flex-end">
              <Button leftSection={<IconUpload size={16} />} loading={indexLoading} onClick={indexPdf}>
                PDF 인덱싱
              </Button>
            </Group>
          </Grid.Col>
        </Grid>
      </Card>

      <Card withBorder radius="lg" p="lg">
        <Group justify="space-between" mb="sm">
          <Text fw={700}>카탈로그 검색</Text>
          <IconDatabaseSearch size={18} />
        </Group>
        <Grid gutter="md">
          <Grid.Col span={{ base: 12, md: 7 }}>
            <TextInput
              label="질문 / 검색어"
              placeholder="예: 16V 3A 어댑터의 출력 스펙과 안전 규격"
              value={queryText}
              onChange={(e) => setQueryText(e.currentTarget.value)}
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 3 }}>
            <TextInput
              label="Model ID 필터(옵션)"
              placeholder="예: NB-ADP-16V"
              value={searchModelId}
              onChange={(e) => setSearchModelId(e.currentTarget.value)}
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 2 }}>
            <NumberInput
              label="Top-K"
              min={1}
              max={50}
              value={topK}
              onChange={(v) => setTopK(typeof v === "number" ? v : 10)}
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 7 }}>
            <TextInput
              label="Part Number 필터(옵션)"
              placeholder="예: ADP-16V-3A"
              value={searchPartNumber}
              onChange={(e) => setSearchPartNumber(e.currentTarget.value)}
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 5 }}>
            <Group justify="flex-end" style={{ height: "100%" }} align="end">
              <Button leftSection={<IconFileSearch size={16} />} loading={searchLoading} onClick={searchCatalog}>
                검색
              </Button>
            </Group>
          </Grid.Col>
        </Grid>
      </Card>

      <Stack gap="sm">
        {results.length === 0 ? (
          <Card withBorder radius="lg" p="lg">
            <Text c="dimmed">카탈로그 검색 결과가 여기에 표시됩니다.</Text>
          </Card>
        ) : (
          results.map((r, idx) => (
            <Card key={`${r.document_id}-${r.page}-${r.chunk_id}-${idx}`} withBorder radius="lg" p="lg">
              <Group justify="space-between" align="flex-start">
                <Stack gap={2}>
                  <Text fw={700}>{r.source || "unknown source"}</Text>
                  <Text size="sm" c="dimmed">
                    page {r.page} · chunk {r.chunk_id}
                  </Text>
                </Stack>
                <Badge variant="light">score {Number(r.score || 0).toFixed(3)}</Badge>
              </Group>
              <Group gap="xs" mt="sm">
                {r.model_id ? <Badge variant="outline">model: {r.model_id}</Badge> : null}
                {r.part_number ? <Badge variant="outline">pn: {r.part_number}</Badge> : null}
                {r.maker ? <Badge variant="outline">maker: {r.maker}</Badge> : null}
              </Group>
              <Text size="sm" mt="sm" style={{ whiteSpace: "pre-wrap" }}>
                {r.text}
              </Text>
            </Card>
          ))
        )}
      </Stack>
    </Stack>
  );
}

