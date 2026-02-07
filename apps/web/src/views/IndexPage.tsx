import React, { useState } from "react";
import {
  Button,
  Card,
  FileInput,
  Grid,
  Group,
  Stack,
  Text,
  TextInput,
  Title,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { IconUpload } from "@tabler/icons-react";

import { useAuth } from "../state/auth";
import { apiFetchJson } from "../utils/api";

export default function IndexPage() {
  const auth = useAuth();
  const [modelId, setModelId] = useState("");
  const [maker, setMaker] = useState("");
  const [partNumber, setPartNumber] = useState("");
  const [category, setCategory] = useState("");
  const [description, setDescription] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);

  async function onIndex() {
    if (!file) {
      notifications.show({ color: "yellow", title: "이미지 필요", message: "인덱싱할 이미지를 선택하세요." });
      return;
    }
    if (!modelId.trim()) {
      notifications.show({ color: "yellow", title: "Model ID 필요", message: "model_id를 입력하세요." });
      return;
    }
    setLoading(true);
    try {
      const form = new FormData();
      form.append("image", file);
      form.append("model_id", modelId.trim());
      form.append("maker", maker);
      form.append("part_number", partNumber);
      form.append("category", category);
      form.append("description", description);

      const headers: Record<string, string> = {};
      if (auth.token) headers.Authorization = `Bearer ${auth.token}`;
      const res = await apiFetchJson<{ status: string }>("/api/v1/hybrid/index", {
        method: "POST",
        headers,
        body: form,
      });
      notifications.show({ color: "teal", title: "인덱싱 완료", message: res.status });
      setFile(null);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      notifications.show({ color: "red", title: "인덱싱 실패", message: msg });
    } finally {
      setLoading(false);
    }
  }

  return (
    <Stack gap="md">
      <Group justify="space-between" align="baseline">
        <Title order={3}>Index Asset</Title>
        <Text c="dimmed" size="sm">
          업로드 → OCR/임베딩 → Milvus 저장
        </Text>
      </Group>

      <Card withBorder radius="lg" p="lg">
        <Grid gutter="md">
          <Grid.Col span={{ base: 12, md: 6 }}>
            <TextInput label="Model ID" placeholder="예: 520d" value={modelId} onChange={(e) => setModelId(e.currentTarget.value)} required />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}>
            <TextInput label="Maker" placeholder="예: BMW" value={maker} onChange={(e) => setMaker(e.currentTarget.value)} />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}>
            <TextInput
              label="Part number"
              placeholder="예: PN-0000"
              value={partNumber}
              onChange={(e) => setPartNumber(e.currentTarget.value)}
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}>
            <TextInput
              label="Category"
              placeholder="예: Headlamp"
              value={category}
              onChange={(e) => setCategory(e.currentTarget.value)}
            />
          </Grid.Col>
          <Grid.Col span={12}>
            <TextInput
              label="Description"
              placeholder="옵션 설명"
              value={description}
              onChange={(e) => setDescription(e.currentTarget.value)}
            />
          </Grid.Col>
          <Grid.Col span={12}>
            <FileInput
              label="Image"
              placeholder="이미지를 선택하세요"
              value={file}
              onChange={setFile}
              accept="image/*"
              required
            />
          </Grid.Col>

          <Grid.Col span={12}>
            <Group justify="flex-end">
              <Button leftSection={<IconUpload size={16} />} loading={loading} onClick={onIndex}>
                인덱싱
              </Button>
            </Group>
          </Grid.Col>
        </Grid>
      </Card>
    </Stack>
  );
}
