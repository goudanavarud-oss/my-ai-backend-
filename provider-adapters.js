const { providers } = require("./providers");

const sleep = (milliseconds) =>
  new Promise((resolve) => setTimeout(resolve, milliseconds));

async function fetchWithTimeout(
  url,
  options = {},
  providerName = "Provider",
  timeoutMs = 60000
) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(url, {
      ...options,
      signal: options.signal
        ? AbortSignal.any([controller.signal, options.signal])
        : controller.signal,
    });
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error(`${providerName} timed out. Choose another provider and retry.`);
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

async function readJsonResponse(response, providerName) {
  const raw = await response.text();
  let payload = {};

  try {
    payload = raw ? JSON.parse(raw) : {};
  } catch {
    payload = {};
  }

  if (!response.ok) {
    throw new Error(
      payload?.detail ||
        payload?.error?.message ||
        payload?.error ||
        `${providerName} returned HTTP ${response.status}.`
    );
  }

  return payload;
}

async function callFal({ prompt, apiKey, signal }) {
  if (!apiKey) {
    throw new Error("Fal.ai requires an API key. Add it in AI Providers.");
  }

  const response = await fetchWithTimeout(
    providers.fal.endpoint,
    {
      method: "POST",
      signal,
      headers: {
        Authorization: `Key ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        prompt,
        image_size: "landscape_4_3",
        num_images: 1,
        enable_safety_checker: true,
      }),
    },
    providers.fal.name,
    120000
  );
  const payload = await readJsonResponse(response, providers.fal.name);
  const imageUrl = payload?.images?.[0]?.url || payload?.image?.url;

  if (!imageUrl) {
    throw new Error("Fal.ai completed but did not return an image URL.");
  }

  return { imageUrl, provider: "fal", model: providers.fal.defaultModel };
}

async function callPollinations({ prompt, onRetry, signal }) {
  const encodedPrompt = encodeURIComponent(prompt.slice(0, 900));
  const imageUrl = providers.pollinations.endpoint.replace(
    "{prompt}",
    encodedPrompt
  );
  const url = `${imageUrl}?nologo=true&enhance=true&width=1080&height=1920`;
  let lastError;
  const maxRetries = 3;

  for (let attempt = 1; attempt <= maxRetries + 1; attempt += 1) {
    const response = await fetchWithTimeout(
      url,
      { signal },
      providers.pollinations.name,
      120000
    );

    if (response.ok) {
      const contentType = response.headers.get("content-type") || "";
      if (!contentType.startsWith("image/")) {
        await response.body?.cancel();
        throw new Error("Pollinations.ai did not return an image.");
      }

      const imageBuffer = Buffer.from(await response.arrayBuffer());
      if (!imageBuffer.length) {
        throw new Error("Pollinations.ai returned an empty image.");
      }
      if (imageBuffer.length > 25 * 1024 * 1024) {
        throw new Error("Pollinations.ai returned an image larger than 25 MB.");
      }
      return {
        imageUrl: url,
        imageBuffer,
        contentType,
        provider: "pollinations",
        model: providers.pollinations.defaultModel,
      };
    }

    await response.body?.cancel();
    lastError = new Error(
      `Pollinations.ai returned HTTP ${response.status} while generating the image.`
    );
    console.error(
      `Pollinations attempt ${attempt} failed:`,
      lastError.message
    );

    const retryable = response.status === 500 || response.status === 502;
    if (retryable && attempt <= maxRetries) {
      await onRetry?.({
        retryAttempt: attempt,
        status: response.status,
        message: "Pollinations.ai is currently busy. Retrying automatically...",
      });
      if (signal?.aborted) throw new Error("Image generation was cancelled.");
      await sleep(3000);
      continue;
    }

    if (!retryable) throw lastError;
    break;
  }

  return {
    imageUrl: "/placeholder-image.svg",
    provider: "pollinations",
    model: providers.pollinations.defaultModel,
    warning:
      lastError?.message ||
      "Pollinations.ai failed after three automatic attempts.",
  };
}

async function callReplicate({ prompt, apiKey, signal }) {
  if (!apiKey) {
    throw new Error("Replicate requires an API key. Add it in AI Providers.");
  }

  const response = await fetchWithTimeout(
    providers.replicate.endpoint,
    {
      method: "POST",
      signal,
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
        Prefer: "wait=5",
      },
      body: JSON.stringify({
        input: {
          prompt,
          num_outputs: 1,
          aspect_ratio: "1:1",
          output_format: "webp",
        },
      }),
    },
    providers.replicate.name
  );
  let prediction = await readJsonResponse(response, providers.replicate.name);

  for (let attempt = 0; attempt < 20; attempt += 1) {
    if (prediction.status === "succeeded") {
      const imageUrl = Array.isArray(prediction.output)
        ? prediction.output[0]
        : prediction.output;
      if (!imageUrl) {
        throw new Error("Replicate completed but did not return an image URL.");
      }
      return {
        imageUrl,
        provider: "replicate",
        model: providers.replicate.defaultModel,
      };
    }

    if (prediction.status === "failed" || prediction.status === "canceled") {
      throw new Error(
        prediction.error ||
          `Replicate generation ${prediction.status}. Choose another provider and retry.`
      );
    }

    const statusUrl = prediction?.urls?.get;
    if (!statusUrl) {
      throw new Error("Replicate did not return a prediction status URL.");
    }

    await sleep(2000);
    if (signal?.aborted) throw new Error("Image generation was cancelled.");
    const statusResponse = await fetchWithTimeout(
      statusUrl,
      { signal, headers: { Authorization: `Bearer ${apiKey}` } },
      providers.replicate.name,
      30000
    );
    prediction = await readJsonResponse(
      statusResponse,
      providers.replicate.name
    );
  }

  throw new Error("Replicate took too long to generate the image.");
}

const imageAdapters = {
  fal: callFal,
  pollinations: callPollinations,
  replicate: callReplicate,
};

async function generateImage(providerId, options) {
  const adapter = imageAdapters[providerId];
  if (!adapter) {
    throw new Error(
      `Provider "${providerId}" does not support image generation in this worker.`
    );
  }
  return adapter(options);
}

async function generateFalVideo({ imageDataUri, prompt, apiKey, signal }) {
  if (!apiKey) throw new Error("Fal.ai video generation requires a Fal.ai API key.");
  const endpoint = "https://queue.fal.run/fal-ai/kling-video/v1/standard/image-to-video";
  const headers = {
    Authorization: `Key ${apiKey}`,
    "Content-Type": "application/json",
  };
  const parseResponse = async (response) => {
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(
        typeof payload.detail === "string"
          ? payload.detail
          : payload.detail?.[0]?.msg || payload.error?.message || `Fal.ai returned HTTP ${response.status}.`
      );
    }
    return payload;
  };
  const submitted = await parseResponse(await fetchWithTimeout(
    endpoint,
    {
      method: "POST",
      headers,
      signal,
      body: JSON.stringify({
        image_url: imageDataUri,
        prompt: String(prompt || "Cinematic subtle motion, natural movement, no text.").slice(0, 1000),
        duration: "5",
      }),
    },
    "Fal.ai Kling Video",
    60000
  ));
  if (!submitted.request_id) {
    // The queue normally returns a request_id, but accept a synchronous response too.
    if (submitted.video?.url) return { videoUrl: submitted.video.url, provider: "fal", model: "fal-ai/kling-video/v1/standard/image-to-video" };
    throw new Error("Fal.ai did not return a video request ID.");
  }
  const requestId = encodeURIComponent(submitted.request_id);
  const requestPrefix = new URL(`${endpoint}/requests/${requestId}/`);
  const queueUrl = (value, suffix) => {
    const url = new URL(value || `${requestPrefix.href}${suffix}`);
    if (url.protocol !== "https:" || url.host !== "queue.fal.run" ||
        url.pathname !== `${requestPrefix.pathname}${suffix}`) {
      throw new Error("Fal.ai returned an invalid queue URL.");
    }
    return url.href;
  };
  const statusUrl = queueUrl(submitted.status_url, "status");
  const responseUrl = queueUrl(submitted.response_url, "response");
  let completed = false;
  try {
    while (!completed) {
      if (signal?.aborted) throw new Error("Fal.ai video generation was cancelled.");
      const status = await parseResponse(await fetchWithTimeout(statusUrl, { headers, signal }, "Fal.ai Kling Video", 30000));
      if (status.status === "COMPLETED") {
        completed = true;
        break;
      }
      if (status.status !== "IN_QUEUE" && status.status !== "IN_PROGRESS") {
        throw new Error(`Fal.ai video generation ${status.status || "failed"}.`);
      }
      await new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
          signal?.removeEventListener("abort", abort);
          resolve();
        }, 2500);
        const abort = () => {
          clearTimeout(timer);
          reject(new Error("Fal.ai video generation was cancelled."));
        };
        signal?.addEventListener("abort", abort, { once: true });
      });
    }
    const result = await parseResponse(await fetchWithTimeout(responseUrl, { headers, signal }, "Fal.ai Kling Video", 30000));
    if (!result.video?.url) throw new Error("Fal.ai completed but returned no video URL.");
    return { videoUrl: result.video.url, provider: "fal", model: "fal-ai/kling-video/v1/standard/image-to-video" };
  } catch (error) {
    // Cancel a queued paid job if the user stopped it or the step timed out.
    if (signal?.aborted) {
      fetch(queueUrl(submitted.cancel_url, "cancel"), {
        method: "PUT",
        headers: { Authorization: `Key ${apiKey}` },
        signal: AbortSignal.timeout(3000),
      }).catch(() => {});
    }
    throw error;
  }
}

module.exports = { generateImage, generateFalVideo, imageAdapters };
  
