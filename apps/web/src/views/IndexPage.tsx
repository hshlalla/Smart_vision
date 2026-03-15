import React, { useEffect, useState } from "react";
import {
  Badge,
  Button,
  Card,
  FileInput,
  Grid,
  Group,
  Image,
  NumberInput,
  Stack,
  Text,
  TextInput,
  Textarea,
  Title,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { IconSparkles, IconUpload } from "@tabler/icons-react";

import { useAuth } from "../state/auth";
import { apiFetchJson, toBase64 } from "../utils/api";

type MetadataDraft = {
  model_id: string;
  maker: string;
  part_number: string;
  category: string;
  description: string;
  product_info: string;
  price_value: number | null;
  source?: string;
};

const EMPTY_DRAFT: MetadataDraft = {
  model_id: "",
  maker: "",
  part_number: "",
  category: "",
  description: "",
  product_info: "",
  price_value: null,
  source: "",
};

export default function IndexPage() {
  const auth = useAuth();
  const [files, setFiles] = useState<File[]>([]);
  const [previewUrls, setPreviewUrls] = useState<string[]>([]);
  const [draft, setDraft] = useState<MetadataDraft>(EMPTY_DRAFT);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [confirmLoading, setConfirmLoading] = useState(false);
  const [encodingPercent, setEncodingPercent] = useState<number | null>(null);

  useEffect(() => {
    if (!files.length) {
      setPreviewUrls([]);
      return;
    }
    const objectUrls = files.map((file) => URL.createObjectURL(file));
    setPreviewUrls(objectUrls);
    return () => objectUrls.forEach((url) => URL.revokeObjectURL(url));
  }, [files]);

  function setDraftField<K extends keyof MetadataDraft>(key: K, value: MetadataDraft[K]) {
    setDraft((prev) => ({ ...prev, [key]: value }));
  }

  async function encodeSelectedFiles(): Promise<string[]> {
    if (!files.length) {
      throw new Error("이미지를 선택하세요.");
    }
    const selected = files.slice(0, 4);
    const encoded: string[] = [];
    for (let index = 0; index < selected.length; index += 1) {
      const encodedImage = await toBase64(selected[index], {
        maxBytes: 5 * 1024 * 1024,
        maxDimension: 1600,
        quality: 0.82,
        onProgress: (pct) => setEncodingPercent(Math.round(((index + pct / 100) / selected.length) * 100)),
      });
      encoded.push(encodedImage);
    }
    return encoded;
  }

  async function runPreview() {
    if (!files.length) {
      notifications.show({ color: "yellow", title: "이미지 필요", message: "메타를 생성할 이미지를 선택하세요." });
      return;
    }
    setPreviewLoading(true);
    try {
      const image_base64_list = await encodeSelectedFiles();
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (auth.token) headers.Authorization = `Bearer ${auth.token}`;
      const res = await apiFetchJson<{ draft: MetadataDraft }>("/api/v1/hybrid/index/preview", {
        method: "POST",
        headers,
        body: JSON.stringify({ image_base64_list }),
      });
      setDraft({
        model_id: res.draft.model_id || "",
        maker: res.draft.maker || "",
        part_number: res.draft.part_number || "",
        category: res.draft.category || "",
        description: res.draft.description || "",
        product_info: res.draft.product_info || "",
        price_value: typeof res.draft.price_value === "number" ? res.draft.price_value : null,
        source: res.draft.source || "openai",
      });
      notifications.show({ color: "teal", title: "초안 생성 완료", message: "생성된 메타데이터를 검토해 주세요." });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      notifications.show({ color: "red", title: "초안 생성 실패", message: msg });
    } finally {
      setPreviewLoading(false);
      setEncodingPercent(null);
    }
  }

  async function onConfirmIndex() {
    if (!files.length) {
      notifications.show({ color: "yellow", title: "이미지 필요", message: "인덱싱할 이미지를 선택하세요." });
      return;
    }
    setConfirmLoading(true);
    try {
      const image_base64_list = await encodeSelectedFiles();
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (auth.token) headers.Authorization = `Bearer ${auth.token}`;
      const res = await apiFetchJson<{ status: string; model_id?: string }>("/api/v1/hybrid/index/confirm", {
        method: "POST",
        headers,
        body: JSON.stringify({
          image_base64: image_base64_list[0],
          image_base64_list,
          ...draft,
        }),
      });
      notifications.show({
        color: "teal",
        title: "인덱싱 완료",
        message: res.model_id ? `${res.status} (${res.model_id})` : res.status,
      });
      if (res.model_id) {
        setDraft((prev) => ({ ...prev, model_id: res.model_id || prev.model_id }));
      }
      setFiles([]);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      notifications.show({ color: "red", title: "인덱싱 실패", message: msg });
    } finally {
      setConfirmLoading(false);
      setEncodingPercent(null);
    }
  }

  return (
    <Stack gap="md">
      <Group justify="space-between" align="baseline">
        <Title order={3}>Index Asset</Title>
        <Text c="dimmed" size="sm">
          업로드 → GPT 초안 생성 → 수정 → 확인 후 임베딩/저장
        </Text>
      </Group>

      <Card withBorder radius="lg" p="lg">
        <Grid gutter="md">
          <Grid.Col span={12}>
            <FileInput
              label="Images"
              placeholder="이미지를 선택하세요"
              value={files}
              onChange={(value) => setFiles(Array.isArray(value) ? value : value ? [value] : [])}
              accept="image/*"
              multiple
              required
            />
          </Grid.Col>

          {previewUrls.length > 0 ? (
            <Grid.Col span={12}>
              <Card withBorder radius="md" p="sm">
                <Group align="flex-start">
                  {previewUrls.slice(0, 4).map((url, index) => (
                    <Image key={url} src={url} alt={files[index]?.name || `image-${index + 1}`} radius="md" h={96} w={96} fit="cover" />
                  ))}
                  <Stack gap={4} style={{ minWidth: 0, flex: 1 }}>
                    <Text fw={600} lineClamp={1}>
                      {files.length}장 선택됨
                    </Text>
                    <Text size="sm" c="dimmed">
                      GPT 메타 초안은 최대 4장을 함께 보고 생성하고, 확인 후 저장 시 선택한 이미지들을 같은 모델로 모두 인덱싱합니다.
                    </Text>
                  </Stack>
                </Group>
              </Card>
            </Grid.Col>
          ) : null}

          <Grid.Col span={12}>
            <Group justify="space-between">
              <Group gap="xs">
                {draft.source ? <Badge variant="light">{draft.source}</Badge> : null}
              </Group>
              <Button leftSection={<IconSparkles size={16} />} loading={previewLoading} onClick={runPreview}>
                메타 자동 추출
              </Button>
            </Group>
            {encodingPercent !== null ? (
              <Text size="xs" c="dimmed" mt={6}>
                이미지 인코딩 중... {encodingPercent}%
              </Text>
            ) : null}
          </Grid.Col>

          <Grid.Col span={{ base: 12, md: 6 }}>
            <TextInput
              label="Model ID"
              placeholder="비우면 confirm 시 자동 할당"
              value={draft.model_id}
              onChange={(e) => setDraftField("model_id", e.currentTarget.value)}
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}>
            <TextInput
              label="Maker"
              placeholder="예: Fuji Electric"
              value={draft.maker}
              onChange={(e) => setDraftField("maker", e.currentTarget.value)}
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}>
            <TextInput
              label="Part number"
              placeholder="예: SC50BAA"
              value={draft.part_number}
              onChange={(e) => setDraftField("part_number", e.currentTarget.value)}
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}>
            <TextInput
              label="Category"
              placeholder="예: magnetic_contactor"
              value={draft.category}
              onChange={(e) => setDraftField("category", e.currentTarget.value)}
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}>
            <TextInput
              label="Product info"
              placeholder="예: magnetic contactor"
              value={draft.product_info}
              onChange={(e) => setDraftField("product_info", e.currentTarget.value)}
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}>
            <NumberInput
              label="Price value"
              placeholder="예상 가격"
              value={draft.price_value ?? undefined}
              onChange={(v) => setDraftField("price_value", typeof v === "number" ? v : null)}
              min={0}
              thousandSeparator=","
            />
          </Grid.Col>
          <Grid.Col span={12}>
            <Textarea
              label="Description"
              placeholder="검색용 설명"
              minRows={3}
              value={draft.description}
              onChange={(e) => setDraftField("description", e.currentTarget.value)}
            />
          </Grid.Col>

          <Grid.Col span={12}>
            <Group justify="flex-end">
              <Button leftSection={<IconUpload size={16} />} loading={confirmLoading} onClick={onConfirmIndex}>
                확인 후 저장
              </Button>
            </Group>
          </Grid.Col>
        </Grid>
      </Card>
    </Stack>
  );
}
