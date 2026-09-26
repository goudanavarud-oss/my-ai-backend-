require("dotenv").config();

const express = require("express");
const path = require("path");
const { createClient } = require("@supabase/supabase-js");
const { providers, getPublicProviders } = require("./providers");
const { outputsDirectory } = require("./media-adapters");
const {
  STEP_TIMEOUT_MS,
  VIDEO_ASSEMBLY_TIMEOUT_MS,
  createRunControl,
  getGroqStepPrompt,
  loadProviderConfigs,
  parseScriptScenes,
  parseStoredOutput,
  pipelineStepNames,
  runPipelineWorker,
  withStepTimeout,
} = require("./worker.js");

const app = express();
const port = process.env.PORT || 3000;
const activeRuns = new Map();

function startRun(options) {
  const { executionId } = options;
  if (activeRuns.get(executionId)?.cancelled) activeRuns.delete(executionId);
  if (activeRuns.has(executionId)) return false;
  const run = createRunControl();
  activeRuns.set(executionId, run);
  runPipelineWorker({ ...options, run })
    .catch((error) => console.error(`Worker crashed for ${executionId}:`, error))
    .finally(() => {
      if (activeRuns.get(executionId) === run) activeRuns.delete(executionId);
    });
  return true;
}

const publicDirectory = path.join(__dirname, "public");
const defaultPipelineDefinition = pipelineStepNames.map((step_name, index) => ({
  step_number: index + 1,
  step_name,
}));

const defaultGroqModel = "openai/gpt-oss-20b";
const supportedGroqModels = [
  "openai/gpt-oss-20b",
  "openai/gpt-oss-120b",
  "qwen/qwen3.6-27b",
  "qwen/qwen3.8-27b",
];
const imageProviderIds = Object.entries(providers)
  .filter(([, provider]) => provider.category === "image")
  .map(([id]) => id);
const providerKeyFields = Object.values(providers)
  .map((provider) => provider.apiKeyField)
  .filter(Boolean);

function normalizeProviderSelections(value) {
  const selections = {};
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return selections;
  }

  for (const stepNumber of [5, 6, 7]) {
    const providerId = String(value[stepNumber] || "").trim();
    if (imageProviderIds.includes(providerId)) {
      selections[stepNumber] = providerId;
    }
  }
  if (["edge_tts", "elevenlabs"].includes(String(value[10] || "").trim())) {
    selections[10] = String(value[10]).trim();
  }

  return selections;
}

function normalizeGroqModel(value) {
  const model = String(value || "").trim();
  return supportedGroqModels.includes(model) ? model : defaultGroqModel;
}

function normalizeSupabaseUrl(value) {
  const trimmedValue = String(value || "").trim().replace(/\/$/, "");

  if (!trimmedValue) {
    return "";
  }

  return /^https?:\/\//i.test(trimmedValue)
    ? trimmedValue
    : `https://${trimmedValue}.supabase.co`;
}

app.locals.supabaseConfig = {
  supabaseUrl: normalizeSupabaseUrl(process.env.SUPABASE_URL),
  supabaseAnonKey: String(process.env.SUPABASE_ANON_KEY || "").trim(),
};

function getSupabaseClient(accessToken) {
  return createClient(
    app.locals.supabaseConfig.supabaseUrl,
    app.locals.supabaseConfig.supabaseAnonKey,
    {
      auth: {
        autoRefreshToken: false,
        persistSession: false,
      },
      global: accessToken
        ? { headers: { Authorization: `Bearer ${accessToken}` } }
        : undefined,
    }
  );
}

async function requireUser(req, res, next) {
  const authorization = req.headers.authorization || "";
  const accessToken = authorization.startsWith("Bearer ")
    ? authorization.slice("Bearer ".length)
    : "";

  if (!accessToken) {
    return res.status(401).json({ error: "A Supabase access token is required." });
  }

  const supabase = getSupabaseClient(accessToken);
  const { data, error } = await supabase.auth.getUser(accessToken);

  if (error || !data.user) {
    return res.status(401).json({ error: "Your session is no longer valid." });
  }

  req.supabase = supabase;
  req.user = data.user;
  next();
}

function getErrorMessage(error) {
  return error?.message || "Something went wrong. Please try again.";
}

app.use(express.json({ limit: "100kb" }));
app.get("/vendor/supabase.js", (req, res) => {
  res.sendFile(
    path.join(
      __dirname,
      "node_modules",
      "@supabase",
      "supabase-js",
      "dist",
      "umd",
      "supabase.js"
    )
  );
});
app.get("/api/outputs/:filename", requireUser, async (req, res) => {
  const filename = path.basename(req.params.filename);
  const match = filename.match(
    /^([0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})-step-(07(?:-[0-9a-f]{12})?\.(?:jpg|png|webp)|08(?:-[0-9a-f]{12})?\.mp4|09(?:-[0-9a-f]{12})?\.mp3|10(?:-[0-9a-f]{12})?\.mp3|13(?:-[0-9a-f]{12})?\.mp4)$/i
  );
  if (!match) {
    return res.status(404).json({ error: "Output file not found." });
  }

  const executionResult = await req.supabase
    .from("executions")
    .select("id")
    .eq("id", match[1])
    .eq("user_id", req.user.id)
    .maybeSingle();

  if (executionResult.error || !executionResult.data) {
    return res.status(404).json({ error: "Output file not found." });
  }

  const filePath = path.join(outputsDirectory, filename);
  res.sendFile(filePath, (error) => {
    if (error && !res.headersSent) {
      res.status(error.statusCode || 404).json({ error: "Output file not found." });
    }
  });
});
app.use(express.static(publicDirectory));

app.get("/api/providers", requireUser, async (req, res) => {
  try {
    const configuredFields = await loadProviderConfigs(
      req.supabase,
      req.user.id
    );
    res.json({ providers: getPublicProviders(configuredFields) });
  } catch (error) {
    return res.status(500).json({ error: getErrorMessage(error) });
  }
});

app.put("/api/provider-configs", requireUser, async (req, res) => {
  const incoming =
    req.body?.configs &&
    typeof req.body.configs === "object" &&
    !Array.isArray(req.body.configs)
      ? req.body.configs
      : null;

  if (!incoming) {
    return res.status(400).json({ error: "Provider configs must be an object." });
  }

  let merged;
  try {
    merged = await loadProviderConfigs(req.supabase, req.user.id);
  } catch (error) {
    return res.status(500).json({ error: getErrorMessage(error) });
  }
  for (const field of providerKeyFields) {
    if (!(field in incoming)) continue;
    const value = incoming[field];
    if (value === null) {
      delete merged[field];
      continue;
    }
    if (typeof value !== "string") {
      return res.status(400).json({ error: `${field} must be a string or null.` });
    }
    const trimmed = value.trim();
    if (trimmed && trimmed.length < 8) {
      return res.status(400).json({ error: `${field} looks too short.` });
    }
    if (trimmed) merged[field] = trimmed;
  }

  if (incoming.groq_api_key === null) {
    const legacyClearResult = await req.supabase
      .from("profiles")
      .update({
        groq_api_key: null,
        updated_at: new Date().toISOString(),
      })
      .eq("user_id", req.user.id);
    if (legacyClearResult.error) {
      return res.status(500).json({ error: getErrorMessage(legacyClearResult.error) });
    }
  }

  const { data, error } = await req.supabase
    .from("provider_configs")
    .upsert(
      {
        user_id: req.user.id,
        configs: merged,
        updated_at: new Date().toISOString(),
      },
      { onConflict: "user_id" }
    )
    .select("configs")
    .single();

  if (error) {
    return res.status(500).json({ error: getErrorMessage(error) });
  }

  res.json({ providers: getPublicProviders(parseStoredOutput(data.configs)) });
});

app.get("/api/profile", requireUser, async (req, res) => {
  const { data, error } = await req.supabase
    .from("profiles")
    .select("user_id, groq_api_key, groq_model")
    .eq("user_id", req.user.id)
    .maybeSingle();

  if (error) {
    return res.status(500).json({ error: getErrorMessage(error) });
  }

  res.json({
    configured: Boolean(data?.groq_api_key),
    model: normalizeGroqModel(data?.groq_model),
  });
});

app.put("/api/profile", requireUser, async (req, res) => {
  const rawKey = typeof req.body?.groqApiKey === "string"
    ? req.body.groqApiKey.trim()
    : "";
  const rawModel = typeof req.body?.groqModel === "string"
    ? req.body.groqModel.trim()
    : "";
  if (rawKey && rawKey.length < 20) {
    return res.status(400).json({ error: "That Groq API key looks too short." });
  }

  if (rawModel && !supportedGroqModels.includes(rawModel)) {
    return res.status(400).json({ error: "That Groq model is not supported." });
  }

  const model = normalizeGroqModel(rawModel);
  const { data, error } = await req.supabase
    .from("profiles")
    .upsert(
      {
        user_id: req.user.id,
        groq_api_key: rawKey || null,
        groq_model: model,
        updated_at: new Date().toISOString(),
      },
      { onConflict: "user_id" }
    )
    .select("user_id, groq_api_key, groq_model")
    .single();

  if (error) {
    return res.status(500).json({ error: getErrorMessage(error) });
  }

  res.json({
    configured: Boolean(data?.groq_api_key),
    model: normalizeGroqModel(data?.groq_model),
  });
});

app.get("/api/pipelines", requireUser, async (req, res) => {
  const { data, error } = await req.supabase
    .from("pipelines")
    .select("id, user_id, name, definition_json, created_at, updated_at")
    .eq("user_id", req.user.id)
    .order("created_at", { ascending: false });

  if (error) {
    return res.status(500).json({ error: getErrorMessage(error) });
  }

  if (!data.length) {
    return res.json({ pipelines: [] });
  }

  const executionsResult = await req.supabase
    .from("executions")
    .select("id, pipeline_id, user_id, status, current_step, created_at, updated_at")
    .eq("user_id", req.user.id)
    .in(
      "pipeline_id",
      data.map((pipeline) => pipeline.id)
    )
    .order("created_at", { ascending: false });

  if (executionsResult.error) {
    return res.status(500).json({ error: getErrorMessage(executionsResult.error) });
  }

  const latestExecutionByPipeline = new Map();
  for (const execution of executionsResult.data) {
    if (!latestExecutionByPipeline.has(execution.pipeline_id)) {
      latestExecutionByPipeline.set(execution.pipeline_id, execution);
    }
  }

  const latestExecutionIds = [...latestExecutionByPipeline.values()].map(
    (execution) => execution.id
  );
  let steps = [];
  if (latestExecutionIds.length) {
    const stepsResult = await req.supabase
      .from("execution_steps")
      .select("id, execution_id, step_number, step_name, status, provider_used, output_data, updated_at")
      .in("execution_id", latestExecutionIds)
      .order("step_number", { ascending: true });
    if (stepsResult.error) {
      return res.status(500).json({ error: getErrorMessage(stepsResult.error) });
    }
    steps = stepsResult.data;
  }

  res.json({
    pipelines: data.map((pipeline) => {
      const latestExecution =
        latestExecutionByPipeline.get(pipeline.id) || null;
      return {
        ...pipeline,
        latest_execution: latestExecution,
        latest_steps: latestExecution
          ? steps.filter((step) => step.execution_id === latestExecution.id)
          : [],
      };
    }),
  });
});

app.post("/api/pipelines", requireUser, async (req, res) => {
  const name = typeof req.body?.name === "string" ? req.body.name.trim() : "";

  if (!name || name.length > 120) {
    return res.status(400).json({ error: "Pipeline name must be between 1 and 120 characters." });
  }

  const { data, error } = await req.supabase
    .from("pipelines")
    .insert({
      user_id: req.user.id,
      name,
      definition_json: defaultPipelineDefinition,
    })
    .select("id, user_id, name, definition_json, created_at, updated_at")
    .single();

  if (error) {
    return res.status(500).json({ error: getErrorMessage(error) });
  }

  res.status(201).json({ pipeline: data });
});

app.post("/api/pipelines/:pipelineId/executions", requireUser, async (req, res) => {
  const { pipelineId } = req.params;
  const providerSelections = normalizeProviderSelections(req.body?.providers);
  const pipelineResult = await req.supabase
    .from("pipelines")
    .select("id, name, definition_json")
    .eq("id", pipelineId)
    .eq("user_id", req.user.id)
    .single();

  if (pipelineResult.error || !pipelineResult.data) {
    return res.status(404).json({ error: "Pipeline not found." });
  }

  const activeExecution = await req.supabase
    .from("executions")
    .select("id")
    .eq("pipeline_id", pipelineId)
    .eq("user_id", req.user.id)
    .in("status", ["pending", "running", "retrying", "paused"])
    .limit(1);
  if (activeExecution.error) {
    return res.status(500).json({ error: getErrorMessage(activeExecution.error) });
  }
  if (activeExecution.data?.length) {
    return res.status(409).json({
      error: "This pipeline has an active or paused run. Retry the step or stop it before starting fresh.",
    });
  }

  const definition = Array.isArray(pipelineResult.data.definition_json)
    ? pipelineResult.data.definition_json
    : defaultPipelineDefinition;
  const steps = definition.slice(0, 20).map((step, index) => ({
    step_number: index + 1,
    step_name: String(
      (index + 1 >= 5 && index + 1 <= 7) ||
        index + 1 === 8 ||
        index + 1 === 9 ||
        index + 1 === 10 ||
        index + 1 === 13
        ? pipelineStepNames[index]
        : step.step_name || pipelineStepNames[index] || `Pipeline step ${index + 1}`
    ),
    status: "pending",
    provider_used:
      index + 1 >= 5 && index + 1 <= 7
        ? providerSelections[index + 1] || "pollinations"
        : index + 1 === 10
          ? providerSelections[10] || "edge_tts"
        : null,
    output_data: {},
  }));

  if (steps.length !== 20) {
    return res.status(400).json({ error: "Pipeline definitions must contain 20 steps." });
  }

  const executionResult = await req.supabase
    .from("executions")
    .insert({
      pipeline_id: pipelineId,
      user_id: req.user.id,
      status: "pending",
      current_step: 0,
    })
    .select("id, pipeline_id, user_id, status, current_step, created_at, updated_at")
    .single();

  if (executionResult.error) {
    return res.status(500).json({ error: getErrorMessage(executionResult.error) });
  }

  const execution = executionResult.data;
  const stepsResult = await req.supabase
    .from("execution_steps")
    .insert(steps.map((step) => ({ ...step, execution_id: execution.id })))
    .select("id, execution_id, step_number, step_name, status, provider_used, output_data, updated_at");

  if (stepsResult.error) {
    await req.supabase.from("executions").delete().eq("id", execution.id).eq("user_id", req.user.id);
    return res.status(500).json({ error: getErrorMessage(stepsResult.error) });
  }

  const started = startRun({
    supabase: req.supabase,
    executionId: execution.id,
    userId: req.user.id,
    steps: stepsResult.data,
  });
  if (!started) {
    return res.status(409).json({ error: "A worker is already active for this execution." });
  }

  res.status(201).json({ execution, steps: stepsResult.data });
});

app.patch(
  "/api/executions/:executionId/steps/:stepNumber/provider",
  requireUser,
  async (req, res) => {
    const stepNumber = Number(req.params.stepNumber);
    const providerId =
      typeof req.body?.provider === "string" ? req.body.provider.trim() : "";

    if (![5, 6, 7, 10].includes(stepNumber)) {
      return res.status(400).json({ error: "Provider selection is only available for Steps 5–7 and 10." });
    }
    if (stepNumber === 10 ? !["edge_tts", "elevenlabs"].includes(providerId) : !imageProviderIds.includes(providerId)) {
      return res.status(400).json({ error: "Choose a supported provider for this step." });
    }

    const executionResult = await req.supabase
      .from("executions")
      .select("id, status")
      .eq("id", req.params.executionId)
      .eq("user_id", req.user.id)
      .single();

    if (executionResult.error || !executionResult.data) {
      return res.status(404).json({ error: "Execution not found." });
    }
    if (executionResult.data.status === "running") {
      return res.status(409).json({ error: "Wait for the current step to finish before changing providers." });
    }

    const { data, error } = await req.supabase
      .from("execution_steps")
      .update({
        provider_used: providerId,
        updated_at: new Date().toISOString(),
      })
      .eq("execution_id", req.params.executionId)
      .eq("step_number", stepNumber)
      .select("id, execution_id, step_number, step_name, status, provider_used, output_data")
      .single();

    if (error) {
      return res.status(500).json({ error: getErrorMessage(error) });
    }

    res.json({ step: data });
  }
);

app.post(
  "/api/executions/:executionId/steps/:stepNumber/retry",
  requireUser,
  async (req, res) => {
    const stepNumber = Number(req.params.stepNumber);
    const providerId =
      typeof req.body?.provider === "string" ? req.body.provider.trim() : "";

    if (!Number.isInteger(stepNumber) || stepNumber < 1 || stepNumber > 20) {
      return res.status(400).json({ error: "Choose a step from 1 to 20." });
    }
    if (providerId && (
      stepNumber === 10
        ? !["edge_tts", "elevenlabs"].includes(providerId)
        : ![5, 6, 7].includes(stepNumber) || !imageProviderIds.includes(providerId)
    )) {
      return res.status(400).json({ error: "Choose a supported provider for this step." });
    }

    const executionResult = await req.supabase
      .from("executions")
      .select("id, status, current_step")
      .eq("id", req.params.executionId)
      .eq("user_id", req.user.id)
      .maybeSingle();
    if (executionResult.error || !executionResult.data) {
      return res.status(404).json({ error: "Execution not found." });
    }
    if (executionResult.data.status === "failed") {
      return res.status(409).json({ error: "This pipeline was stopped. Start a fresh run instead." });
    }

    const stepsResult = await req.supabase
      .from("execution_steps")
      .select("id, execution_id, step_number, step_name, status, provider_used, output_data, updated_at")
      .eq("execution_id", req.params.executionId)
      .order("step_number", { ascending: true });

    if (stepsResult.error) {
      return res.status(500).json({ error: getErrorMessage(stepsResult.error) });
    }

    const target = stepsResult.data.find((step) => step.step_number === stepNumber);
    const retryTimeoutMs = stepNumber === 13 ? VIDEO_ASSEMBLY_TIMEOUT_MS : STEP_TIMEOUT_MS;
    const stuck = target?.status === "processing" &&
      Date.now() - new Date(target.updated_at).getTime() >= retryTimeoutMs;
    if (!target || (target.status !== "failed" && !stuck)) {
      return res.status(409).json({ error: `Only a failed step or a step processing for over ${retryTimeoutMs / 60000} minutes can be retried.` });
    }
    if (
      stepsResult.data.some(
        (step) =>
          step.step_number < stepNumber &&
          !["completed", "completed_with_warning"].includes(step.status)
      )
    ) {
      return res.status(409).json({ error: "A previous step has not completed. Retry that step first." });
    }
    if (
      stepsResult.data.some(
        (step) =>
          step.step_number > stepNumber && step.status !== "pending"
      )
    ) {
      return res.status(409).json({ error: "Later steps have already started; this step cannot be retried safely." });
    }
    if (!stuck && activeRuns.has(req.params.executionId)) {
      return res.status(409).json({ error: "The previous worker is still closing. Retry in a moment." });
    }

    const claimExecution = await req.supabase
      .from("executions")
      .update({
        status: "retrying",
        current_step: stepNumber,
        updated_at: new Date().toISOString(),
      })
      .eq("id", req.params.executionId)
      .eq("user_id", req.user.id)
      .eq("status", stuck ? "running" : "paused")
      .select("id")
      .maybeSingle();

   
