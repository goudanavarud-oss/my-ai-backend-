const { randomBytes } = require("crypto");
const { providers } = require("./providers");
const { generateImage, generateFalVideo } = require("./provider-adapters");
const {
  assembleVideo,
  createKenBurnsClip,
  ensureOutputsDirectory,
  localImageDataUri,
  localPathFromOutputUrl,
  persistFalVideo,
  persistImage,
  synthesizeElevenLabs,
  synthesizeSpeech,
} = require("./media-adapters");

const STEP_TIMEOUT_MS = 3 * 60 * 1000;
const VIDEO_ASSEMBLY_TIMEOUT_MS = 5 * 60 * 1000;

const pipelineStepNames = [
  "Define the outcome",
  "Audience DNA",
  "Competitor Analysis",
  "First Draft",
  "Creative Direction",
  "Visual Prompt",
  "Visual Generation",
  "Image-to-Video",
  "Voice Synthesis",
  "Voiceover",
  "Capture the insights",
  "Make the key changes",
  "Video Assembly",
  "Prepare the launch",
  "Tell the right people",
  "Release the work",
  "Watch the signals",
  "Document the learnings",
  "Celebrate the progress",
  "Choose what is next",
];

const imageProviderIds = Object.entries(providers)
  .filter(([, provider]) => provider.category === "image")
  .map(([id]) => id);

const providerKeyFields = Object.values(providers)
  .map((provider) => provider.apiKeyField)
  .filter(Boolean);

function cancelledError() {
  return new Error("Manually stopped by user.");
}

function checkRun(run) {
  if (run.cancelled) throw cancelledError();
}

async function withStepTimeout(
  work,
  run,
  milliseconds = STEP_TIMEOUT_MS,
  timeoutMessage = "Step timed out after 3 minutes"
) {
  const controller = new AbortController();
  run.controller = controller;
  let timer;
  const timedOut = new Promise((_, reject) => {
    timer = setTimeout(() => {
      reject(new Error(timeoutMessage));
      controller.abort();
    }, milliseconds);
  });
  try {
    return await Promise.race([work(controller.signal), timedOut, run.stopped]);
  } finally {
    clearTimeout(timer);
    if (run.controller === controller) run.controller = null;
  }
}

function createRunControl() {
  let rejectStopped;
  const run = {
    cancelled: false,
    attemptId: randomBytes(6).toString("hex"),
    controller: null,
    stopped: new Promise((_, reject) => {
      rejectStopped = reject;
    }),
    stop() {
      if (this.cancelled) return;
      this.cancelled = true;
      this.controller?.abort();
      rejectStopped(cancelledError());
    },
  };
  run.stopped.catch(() => {});
  return run;
}

function getErrorMessage(error) {
  return error?.message || "Something went wrong. Please try again.";
}

async function generateGroqText(
  apiKey,
  model,
  { systemPrompt, userPrompt, temperature = 0.7, maxTokens = 500 },
  signal,
  contentOnly = false
) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);

  try {
    const response = await fetch(providers.groq.endpoint, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model,
        temperature,
        max_tokens: maxTokens,
        messages: [
          {
            role: "system",
            content: systemPrompt,
          },
          {
            role: "user",
            content: userPrompt,
          },
        ],
      }),
      signal: signal ? AbortSignal.any([controller.signal, signal]) : controller.signal,
    });

    const rawResponse = await response.text();
    console.log("Groq raw response:", rawResponse);

    let payload = {};
    try {
      payload = JSON.parse(rawResponse);
    } catch {
      payload = {};
    }

    if (response.status === 429) {
      throw new Error("Groq API rate limit exceeded. Please wait a minute and try again.");
    }

    if (!response.ok) {
      throw new Error(
        payload?.error?.message || `Groq returned HTTP ${response.status}.`
      );
    }

    const choice = payload?.choices?.[0];
    const message = choice?.message || {};
    const getText = (value) => {
      if (typeof value === "string") return value.trim();
      if (Array.isArray(value)) {
        return value
          .map((part) => (typeof part === "string" ? part : part?.text || ""))
          .join("")
          .trim();
      }
      return "";
    };
    const text = contentOnly
      ? getText(message.content)
      : getText(message.content) ||
        getText(message.reasoning) ||
        getText(choice?.text);

    if (!text) {
      throw new Error("Groq returned a valid JSON but the text content was empty.");
    }

    return text;
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error("Groq took too long to respond. Please try again.");
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

function parseStoredOutput(value) {
  if (value && typeof value === "object" && !Array.isArray(value)) return value;
  if (typeof value !== "string") return {};

  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? parsed
      : {};
  } catch {
    return {};
  }
}

async function loadProviderConfigs(supabase, userId) {
  const configResult = await supabase
    .from("provider_configs")
    .select("configs")
    .eq("user_id", userId)
    .maybeSingle();
  if (configResult.error) throw configResult.error;

  const configs = parseStoredOutput(configResult.data?.configs);
  if (configs.groq_api_key) return configs;

  const legacyResult = await supabase
    .from("profiles")
    .select("groq_api_key")
    .eq("user_id", userId)
    .maybeSingle();
  if (legacyResult.error) throw legacyResult.error;
  if (!legacyResult.data?.groq_api_key) return configs;

  configs.groq_api_key = legacyResult.data.groq_api_key;
  const migrationResult = await supabase.from("provider_configs").upsert(
    {
      user_id: userId,
      configs,
      updated_at: new Date().toISOString(),
    },
    { onConflict: "user_id" }
  );
  if (migrationResult.error) throw migrationResult.error;

  const legacyClearResult = await supabase
    .from("profiles")
    .update({
      groq_api_key: null,
      updated_at: new Date().toISOString(),
    })
    .eq("user_id", userId);
  if (legacyClearResult.error) throw legacyClearResult.error;

  return configs;
}

function buildVisualPrompt(previousOutputs) {
  const script = previousOutputs[4] || previousOutputs[1] || "";
  const audience = previousOutputs[2] || "";
  const competitors = previousOutputs[3] || "";
  return [
    "Create a polished editorial campaign image for this concept.",
    `Script: ${script}`,
    `Audience: ${audience}`,
    `Positioning: ${competitors}`,
    "No text, captions, logos, watermarks, or UI elements in the image.",
  ]
    .join("\n")
    .slice(0, 1800);
}

function parseScriptScenes(raw) {
  const source = String(raw || "").trim().replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "");
  let scenes;
  try {
    scenes = JSON.parse(source);
  } catch {
    throw new Error("Step 4 did not return valid scene JSON. Retry Step 4.");
  }
  if (!Array.isArray(scenes) || scenes.length < 3 || scenes.length > 8) {
    throw new Error("Step 4 must return a JSON array of 3 to 8 scenes. Retry Step 4.");
  }
  return scenes.map((scene, index) => {
    if (!scene || typeof scene !== "object" || Array.isArray(scene) ||
      scene.scene_number !== index + 1 ||
      typeof scene.duration_seconds !== "number" ||
      !Number.isFinite(scene.duration_seconds) ||
      scene.duration_seconds < 1 || scene.duration_seconds > 90 ||
      !["dialogue", "visual_prompt", "emotion"].every(
        (field) => typeof scene[field] === "string" && scene[field].trim()
      )) {
      throw new Error(`Step 4 returned an invalid scene ${index + 1}. Retry Step 4.`);
    }
    return {
      scene_number: scene.scene_number,
      duration_seconds: scene.duration_seconds,
      dialogue: scene.dialogue.trim(),
      visual_prompt: scene.visual_prompt.trim(),
      emotion: scene.emotion.trim(),
    };
  });
}

function getGroqStepPrompt(stepNumber, previousOutputs) {
  const prompts = {
    1: {
      systemPrompt: "You are a creative writer.",
      userPrompt:
        "Write exactly 50 words about Ancient India. Do not output any JSON, just the plain text story.",
      temperature: 0.7,
      maxTokens: 120,
    },
    2: {
      systemPrompt:
        "You are a sharp audience researcher and content strategist. Return plain text only, never JSON.",
      userPrompt: `Create an Audience DNA analysis based only on the Step 1 context below. Identify the target audience, their motivations, needs, frustrations, likely objections, desired transformation, and the tone and message most likely to resonate. Use clear headings and concise bullets. Do not discuss these instructions.

Step 1 output:
${previousOutputs[1]}`,
      temperature: 0.4,
      maxTokens: 500,
    },
    3: {
      systemPrompt:
        "You are a practical competitive strategist. Return plain text only, never JSON.",
      userPrompt: `Create a Competitor Analysis using the Step 1 concept and Step 2 Audience DNA below. Find three realistic competitor angles or alternative approaches, compare their strengths and weaknesses, and end with differentiated positioning opportunities. Use clear headings and concise bullets. Do not invent specific company facts; when the concept is broad, describe competitor types or alternatives instead. Do not discuss these instructions.

Step 1 output:
${previousOutputs[1]}

Step 2 output:
${previousOutputs[2]}`,
      temperature: 0.4,
      maxTokens: 650,
    },
    4: {
      systemPrompt:
        "You are an experienced video scriptwriter and storyboard editor. Return only a valid JSON array, with no markdown fences or commentary.",
      userPrompt: `Write a video script storyboard using all three outputs below. Return ONLY a JSON array of 4 to 6 scenes. Each object MUST have exactly these fields: "scene_number" (consecutive integers starting at 1), "duration_seconds" (a positive number reflecting the speaking time), "dialogue" (spoken narration), "visual_prompt" (a specific description of what should be shown, with no text or logos in the image), and "emotion" (a short mood or delivery cue). Aim for approximately 100 spoken words across the scenes. Open with a strong hook, build a clear middle, and finish with a natural call to action. Keep dialogue natural and durations realistic for the spoken words. Do not return a wrapper object, notes, titles, code fences, or anything outside the JSON array.

Step 1 output:
${previousOutputs[1]}

Step 2 output:
${previousOutputs[2]}

Step 3 output:
${previousOutputs[3]}`,
      temperature: 0.7,
      maxTokens: 2400,
    },
  };

  return prompts[stepNumber];
      }async function runPipelineWorker({
  supabase,
  executionId,
  userId,
  steps,
  startStep = 1,
  run,
}) {
  let activeStepNumber = startStep;
  const previousOutputs = {};
  const previousStepData = {};

  try {
    await ensureOutputsDirectory();
    for (const step of steps) {
      if (step.step_number >= startStep) continue;
      const output = parseStoredOutput(step.output_data);
      previousStepData[step.step_number] = output;
      if (typeof output.text === "string" && output.text.trim()) {
        previousOutputs[step.step_number] = output.text.trim();
      }
    }

    const providerConfigs = await loadProviderConfigs(supabase, userId);

    checkRun(run);
    const executionStart = await supabase
      .from("executions")
      .update({
        status: "running",
        current_step: Math.max(0, startStep - 1),
        updated_at: new Date().toISOString(),
      })
      .eq("id", executionId)
      .eq("user_id", userId)
      .in("status", ["pending", "running", "retrying"])
      .select("id")
      .maybeSingle();
    if (executionStart.error) throw executionStart.error;
    if (!executionStart.data) {
      checkRun(run);
      throw new Error("Execution is no longer available to run.");
    }

    for (const step of steps) {
      if (step.step_number < startStep) continue;
      checkRun(run);
      activeStepNumber = step.step_number;
      const processing = await supabase
        .from("execution_steps")
        .update({
          status: "processing",
          output_data:
            step.step_number <= 4
              ? { message: `Calling Groq for step ${step.step_number}.` }
              : step.step_number === 7
                ? {
                    message: `Calling ${
                      providers[step.provider_used || "pollinations"]?.name ||
                      step.provider_used
                    } for visual generation.`,
                  }
                : step.step_number === 8
                  ? { message: "Creating a five-second video from the Step 7 image." }
                  : step.step_number === 9
                    ? { message: "Synthesizing narration with Edge TTS." }
                    : step.step_number === 10
                      ? { message: "Creating the selected voiceover." }
                      : step.step_number === 13
                        ? { message: "Assembling the final video with FFmpeg." }
                        : { message: "Worker is processing this step." },
          updated_at: new Date().toISOString(),
        })
        .eq("execution_id", executionId)
        .eq("step_number", step.step_number)
        .eq("status", "pending")
        .select("id")
        .maybeSingle();

      if (processing.error) throw processing.error;
      if (!processing.data) throw new Error(`Step ${step.step_number} was not pending.`);
      checkRun(run);

      const currentStepUpdate = await supabase
        .from("executions")
        .update({ current_step: step.step_number, updated_at: new Date().toISOString() })
        .eq("id", executionId)
        .eq("user_id", userId);

      if (currentStepUpdate.error) throw currentStepUpdate.error;

      const { outputData, completionStatus } = await withStepTimeout(async (signal) => {
        let outputData;
        let completionStatus = "completed";
        if (step.step_number <= 4) {
          if (!providerConfigs.groq_api_key) {
            throw new Error("Add your Groq API key in Settings before running the AI steps.");
          }

          const model = providers.groq.defaultModel;
          const prompt = getGroqStepPrompt(step.step_number, previousOutputs);
          const text = await generateGroqText(
            providerConfigs.groq_api_key,
            model,
            prompt,
            signal,
            step.step_number === 4
          );
          if (step.step_number === 4) {
            const scenes = parseScriptScenes(text);
            const dialogue = scenes.map((scene) => scene.dialogue).join("\n\n");
            outputData = { scenes, text: dialogue };
            previousOutputs[4] = dialogue;
          } else {
            outputData = { text };
            previousOutputs[step.step_number] = text;
          }
        } else if (step.step_number === 7) {
          const providerId = step.provider_used || "pollinations";
          const provider = providers[providerId];
          if (!provider || provider.category !== "image") {
            throw new Error(
              `Provider "${providerId}" is not a supported image provider.`
            );
          }

          const prompt = buildVisualPrompt(previousOutputs);
          const result = await generateImage(providerId, {
            prompt,
            signal,
            apiKey: provider.apiKeyField
              ? providerConfigs[provider.apiKeyField]
              : null,
            onRetry: async ({ message }) => {
              if (signal.aborted || run.cancelled) return;
              const retryUpdate = await supabase
                .from("execution_steps")
                .update({
                  output_data: { message },
                  updated_at: new Date().toISOString(),
                })
                .eq("execution_id", executionId)
                .eq("step_number", step.step_number)
                .eq("status", "processing");
              if (retryUpdate.error) throw retryUpdate.error;
            },
          });
          const savedImage = await persistImage({
            imageUrl: result.imageUrl,
            imageBuffer: result.imageBuffer,
            contentType: result.contentType,
            executionId,
            attemptId: run.attemptId,
            signal,
          });
          outputData = {
            text: `Image generated with ${provider.name}.`,
            image_url: savedImage.url,
            prompt,
            provider: result.provider,
            model: result.model,
          };
          if (result.warning) {
            completionStatus = "completed_with_warning";
            outputData.text =
              "Pollinations.ai could not generate the visual, so a placeholder image was used.";
            outputData.warning = result.warning;
          }
          previousOutputs[step.step_number] = outputData.text;
        } else if (step.step_number === 8) {
          const imageUrl = previousStepData[7]?.image_url;
          if (!imageUrl) throw new Error("Step 8 needs the generated image from Step 7.");
          let clip;
          let provider;
          if (providerConfigs.fal_api_key && imageUrl !== "/placeholder-image.svg") {
            const imageDataUri = await localImageDataUri(imageUrl, executionId);
            const result = await generateFalVideo({
              imageDataUri,
              prompt: `${previousOutputs[4] || previousOutputs[1] || "Cinematic visual"}. Subtle natural camera motion, no text or logos.`,
              apiKey: providerConfigs.fal_api_key,
              signal,
            });
            clip = await persistFalVideo({
              videoUrl: result.videoUrl,
              executionId,
              attemptId: run.attemptId,
              signal,
            });
            provider = "fal";
          } else {
            clip = await createKenBurnsClip({
              imageUrl,
              executionId,
              attemptId: run.attemptId,
              signal,
            });
            provider = "ffmpeg";
          }
          outputData = {
            text: provider === "fal" ? "Generated a five-second Kling image-to-video clip." : "Created a five-second Ken Burns video clip.",
            video_url: clip.url,
            duration_seconds: 5,
            provider,
          };
        } else if (step.step_number === 9) {
          const audio = await synthesizeSpeech({
            text: previousOutputs[4],
            executionId,
            attemptId: run.attemptId,
            signal,
          });
          outputData = {
            audio_url: audio.url,
            voice: audio.voice,
            provider: "edge_tts",
          };
        } else if (step.step_number === 10) {
          const requestedProvider = step.provider_used || "edge_tts";
          let audio;
          if (requestedProvider === "elevenlabs" && providerConfigs.elevenlabs_api_key) {
            audio = await synthesizeElevenLabs({
              text: previousOutputs[4],
              apiKey: providerConfigs.elevenlabs_api_key,
              executionId,
              attemptId: run.attemptId,
              signal,
            });
          } else {
            audio = await synthesizeSpeech({
              text: previousOutputs[4],
              executionId,
              attemptId: run.attemptId,
              stepNumber: 10,
              signal,
            });
            if (requestedProvider === "elevenlabs") {
              completionStatus = "completed_with_warning";
            }
          }
          outputData = {
            text: `Voiceover generated with ${audio.provider === "elevenlabs" ? "ElevenLabs" : "Edge TTS"}.`,
            audio_url: audio.url,
            voice: audio.voice,
            provider: audio.provider || "edge_tts",
            ...(completionStatus === "completed_with_warning" ? { warning: "ElevenLabs key not set; used Edge TTS." } : {}),
          };
        } else if (step.step_number === 13) {
          const imageUrl = previousStepData[7]?.image_url;
          const audioPath = localPathFromOutputUrl(previousStepData[10]?.audio_url);
          let video;
          for (let attempt = 0; attempt < 2; attempt += 1) {
            try {
              video = await assembleVideo({
                imageUrl,
                audioPath,
                executionId,
                attemptId: run.attemptId,
                signal,
                attempt: attempt + 1,
              });
              break;
            } catch (error) {
              if (signal.aborted || attempt === 1 || error.message === "Video assembly timed out.") throw error;
              console.warn(`Step 13 attempt 1 failed; retrying in 10 seconds:`, error.message);
              await new Promise((resolve, reject) => {
                const onAbort = () => {
                  clearTimeout(timer);
                  reject(new Error("Video assembly timed out."));
                };
                const timer = setTimeout(() => {
                  signal.removeEventListener("abort", onAbort);
                  resolve();
                }, 10000);
                signal.addEventListener("abort", onAbort, { once: true });
                if (signal.aborted) onAbort();
              });
            }
          }
          outputData = {
            video_url: video.url,
            duration_seconds: Number(video.duration.toFixed(2)),
          };
        } else {
          await new Promise((resolve, reject) => {
            const timer = setTimeout(resolve, 2000);
            signal.addEventListener("abort", () => {
              clearTimeout(timer);
              reject(new Error("Step timed out after 3 minutes"));
            }, { once: true });
          });
          outputData = {
            message: "Step completed by the workflow worker.",
            completed_at: new Date().toISOString(),
          };
        }
        return { outputData, completionStatus };
      }, run, step.step_number === 13 ? VIDEO_ASSEMBLY_TIMEOUT_MS : STEP_TIMEOUT_MS, step.step_number === 13 ? "Video assembly timed out." : "Step timed out after 3 minutes");
      checkRun(run);
      previousStepData[step.step_number] = outputData;

      const completed = await supabase
        .from("execution_steps")
        .update({
          status: completionStatus,
          output_data: outputData,
          updated_at: new Date().toISOString(),
        })
        .eq("execution_id", executionId)
        .eq("step_number", step.step_number)
        .eq("status", "processing")
        .select("id")
        .maybeSingle();

      if (completed.error) throw completed.error;
      if (!completed.data) {
        checkRun(run);
        throw new Error(`Step ${step.step_number} was changed before it completed.`);
      }
    }

    checkRun(run);
    const executionComplete = await supabase
      .from("executions")
      .update({
        status: "completed",
        current_step: steps.length,
        updated_at: new Date().toISOString(),
      })
      .eq("id", executionId)
      .eq("user_id", userId)
      .eq("status", "running");

    if (executionComplete.error) throw executionComplete.error;
  } catch (error) {
    if (run.cancelled) return;
    const safeMessage = getErrorMessage(error);
    console.error(`Worker failed for execution ${executionId}:`, safeMessage);

    const pause = await supabase
      .from("executions")
      .update({
        status: "paused",
        current_step: activeStepNumber,
        updated_at: new Date().toISOString(),
      })
      .eq("id", executionId)
      .eq("user_id", userId)
      .eq("status", "running")
      .select("id")
      .maybeSingle();
    if (pause.error) console.error("Could not pause execution:", pause.error);
    if (!pause.data) return;

    if (activeStepNumber > 0) {
      const failed = await supabase
        .from("execution_steps")
        .update({
          status: "failed",
          output_data: { error: safeMessage },
          updated_at: new Date().toISOString(),
        })
        .eq("execution_id", executionId)
        .eq("step_number", activeStepNumber)
        .in("status", ["processing", "pending"]);
      if (failed.error) console.error("Could not save step failure:", failed.error);
    }
  }
}

module.exports = {
  STEP_TIMEOUT_MS,
  VIDEO_ASSEMBLY_TIMEOUT_MS,
  createRunControl,
  getGroqStepPrompt,
  loadProviderConfigs,
  parseScriptScenes,
  parseStoredOutput,
  pipelineStepNames,
  providerKeyFields,
  imageProviderIds,
  runPipelineWorker,
  withStepTimeout,
};
