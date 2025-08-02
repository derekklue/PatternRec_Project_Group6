import { BertWordPieceTokenizer } from "@huggingface/tokenizers";

const initTokenizer = async () => {
  // you’ll need the vocab.json from your HF model
  window.tokenizer = await BertWordPieceTokenizer.fromOptions({
    vocabFile: "./models/vocab.txt",
    lowercase: true
  });
};

const encodeText = async (text) => {
  const encode = await window.tokenizer.encode(text);
  // pad/truncate to your maxLen (e.g. 20)
  const ids   = encode.ids.slice(0, maxLen);
  const mask  = encode.attentionMask.slice(0, maxLen);
  while (ids.length < maxLen) { ids.push(0); mask.push(0); }
  return { ids, mask };
};

import { initTokenizer, encodeText } from "models/tokenizer.js";  // from step 2

  let session;
  window.addEventListener('DOMContentLoaded', async () => {
    await initTokenizer();
    session = await ort.InferenceSession.create("models/bert_recipe_model.onnx");
  });

  async function getEmbedding(text) {
    const { ids, mask } = await encodeText(text);
    const inputIds     = new ort.Tensor("int64", BigInt64Array.from(ids.map(x=>BigInt(x))), [1, ids.length]);
    const attentionMask= new ort.Tensor("int64", BigInt64Array.from(mask.map(x=>BigInt(x))), [1, mask.length]);

    const feeds = { input_ids: inputIds, attention_mask: attentionMask };
    const results = await session.run(feeds);
    return results.embeddings.data;  // Float32Array of size [embedding_dim]
  }
