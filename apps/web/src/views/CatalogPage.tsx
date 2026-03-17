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
  Switch,
  Text,
  TextInput,
  Title,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { IconDatabaseSearch, IconFileSearch, IconFileTypePdf, IconUpload } from "@tabler/icons-react";

import MarkdownBlocks from "../components/MarkdownBlocks";
import { useAuth } from "../state/auth";
import { useI18n } from "../state/i18n";
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
  const { t } = useI18n();

  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [source, setSource] = useState("");
  const [indexModelId, setIndexModelId] = useState("");
  const [indexPartNumber, setIndexPartNumber] = useState("");
  const [indexMaker, setIndexMaker] = useState("");
  const [usePaddleOcr, setUsePaddleOcr] = useState(false);
  const [indexLoading, setIndexLoading] = useState(false);

  const [queryText, setQueryText] = useState("");
  const [searchModelId, setSearchModelId] = useState("");
  const [searchPartNumber, setSearchPartNumber] = useState("");
  const [topK, setTopK] = useState<number>(10);
  const [searchLoading, setSearchLoading] = useState(false);
  const [results, setResults] = useState<CatalogSearchResult[]>([]);

  async function indexPdf() {
    if (!pdfFile) {
      notifications.show({ color: "yellow", title: t("catalog.pdfNeededTitle"), message: t("catalog.pdfNeededMessage") });
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
      form.append("use_paddle_ocr", usePaddleOcr ? "true" : "false");

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
        title: t("catalog.indexCompletedTitle"),
        message: t("catalog.indexCompletedMessage", {
          source: res.source,
          pages: res.pages_indexed,
          chunks: res.chunks_indexed,
        }),
      });
      setPdfFile(null);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      notifications.show({ color: "red", title: t("catalog.indexFailedTitle"), message: msg });
    } finally {
      setIndexLoading(false);
    }
  }

  async function searchCatalog() {
    if (!queryText.trim()) {
      notifications.show({ color: "yellow", title: t("catalog.queryNeededTitle"), message: t("catalog.queryNeededMessage") });
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
      notifications.show({
        color: "teal",
        title: t("catalog.searchCompletedTitle"),
        message: t("catalog.searchCompletedMessage", { count: res.results?.length || 0 }),
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      notifications.show({ color: "red", title: t("catalog.searchFailedTitle"), message: msg });
    } finally {
      setSearchLoading(false);
    }
  }

  return (
    <Stack gap="md">
      <Group justify="space-between" align="baseline">
        <Title order={3}>{t("catalog.title")}</Title>
        <Badge variant="light">{t("catalog.badge")}</Badge>
      </Group>

      <Card withBorder radius="lg" p="lg">
        <Group justify="space-between" mb="sm">
          <Text fw={700}>{t("catalog.pdfIndexTitle")}</Text>
          <IconFileTypePdf size={18} />
        </Group>

        <Grid gutter="md">
          <Grid.Col span={12}>
            <FileInput
              label={t("catalog.pdfLabel")}
              placeholder={t("catalog.pdfPlaceholder")}
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
              label={t("catalog.sourceLabel")}
              placeholder={t("catalog.sourcePlaceholder")}
              value={source}
              onChange={(e) => setSource(e.currentTarget.value)}
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}>
            <TextInput
              label={t("catalog.makerLabel")}
              placeholder={t("catalog.makerPlaceholder")}
              value={indexMaker}
              onChange={(e) => setIndexMaker(e.currentTarget.value)}
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}>
            <TextInput
              label={t("catalog.modelIdLabel")}
              placeholder={t("catalog.modelIdPlaceholder")}
              value={indexModelId}
              onChange={(e) => setIndexModelId(e.currentTarget.value)}
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}>
            <TextInput
              label={t("catalog.partNumberLabel")}
              placeholder={t("catalog.partNumberPlaceholder")}
              value={indexPartNumber}
              onChange={(e) => setIndexPartNumber(e.currentTarget.value)}
            />
          </Grid.Col>
          <Grid.Col span={12}>
            <Switch
              checked={usePaddleOcr}
              onChange={(e) => setUsePaddleOcr(e.currentTarget.checked)}
              label={t("catalog.usePaddleLabel")}
              description={t("catalog.usePaddleDescription")}
            />
          </Grid.Col>
          <Grid.Col span={12}>
            <Group justify="flex-end">
              <Button leftSection={<IconUpload size={16} />} loading={indexLoading} onClick={indexPdf}>
                {t("catalog.indexButton")}
              </Button>
            </Group>
          </Grid.Col>
        </Grid>
      </Card>

      <Card withBorder radius="lg" p="lg">
        <Group justify="space-between" mb="sm">
          <Text fw={700}>{t("catalog.searchTitle")}</Text>
          <IconDatabaseSearch size={18} />
        </Group>
        <Grid gutter="md">
          <Grid.Col span={{ base: 12, md: 7 }}>
            <TextInput
              label={t("catalog.queryLabel")}
              placeholder={t("catalog.queryPlaceholder")}
              value={queryText}
              onChange={(e) => setQueryText(e.currentTarget.value)}
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 3 }}>
            <TextInput
              label={t("catalog.modelIdFilterLabel")}
              placeholder={t("catalog.modelIdPlaceholder")}
              value={searchModelId}
              onChange={(e) => setSearchModelId(e.currentTarget.value)}
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 2 }}>
            <NumberInput
              label={t("catalog.topKLabel")}
              min={1}
              max={50}
              value={topK}
              onChange={(v) => setTopK(typeof v === "number" ? v : 10)}
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 7 }}>
            <TextInput
              label={t("catalog.partNumberFilterLabel")}
              placeholder={t("catalog.partNumberPlaceholder")}
              value={searchPartNumber}
              onChange={(e) => setSearchPartNumber(e.currentTarget.value)}
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 5 }}>
            <Group justify="flex-end" style={{ height: "100%" }} align="end">
              <Button leftSection={<IconFileSearch size={16} />} loading={searchLoading} onClick={searchCatalog}>
                {t("catalog.searchButton")}
              </Button>
            </Group>
          </Grid.Col>
        </Grid>
      </Card>

      <Stack gap="sm">
        {results.length === 0 ? (
          <Card withBorder radius="lg" p="lg">
            <Text c="dimmed">{t("catalog.noResults")}</Text>
          </Card>
        ) : (
          results.map((r, idx) => (
            <Card key={`${r.document_id}-${r.page}-${r.chunk_id}-${idx}`} withBorder radius="lg" p="lg">
              <Group justify="space-between" align="flex-start">
                <Stack gap={2}>
                  <Text fw={700}>{r.source || t("catalog.unknownSource")}</Text>
                  <Text size="sm" c="dimmed">
                    {t("catalog.pageChunkLabel", { page: r.page, chunk: r.chunk_id })}
                  </Text>
                </Stack>
                <Badge variant="light">{t("catalog.scoreBadge", { score: Number(r.score || 0).toFixed(3) })}</Badge>
              </Group>
              <Group gap="xs" mt="sm">
                {r.model_id ? <Badge variant="outline">{t("catalog.modelBadge", { value: r.model_id })}</Badge> : null}
                {r.part_number ? <Badge variant="outline">{t("catalog.pnBadge", { value: r.part_number })}</Badge> : null}
                {r.maker ? <Badge variant="outline">{t("catalog.makerBadge", { value: r.maker })}</Badge> : null}
              </Group>
              <Stack gap="xs" mt="sm">
                <MarkdownBlocks content={r.text} />
              </Stack>
            </Card>
          ))
        )}
      </Stack>
    </Stack>
  );
}
