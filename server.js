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
});require("dotenv").config();

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
