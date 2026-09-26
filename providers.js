const providers = {
  groq: {
    name: "Groq",
    category: "text",
    isFree: true,
    apiKeyField: "groq_api_key",
    endpoint: "https://api.groq.com/openai/v1/chat/completions",
    defaultModel: "openai/gpt-oss-20b",
  },
  fal: {
    name: "Fal.ai",
    category: "image",
    isFree: false,
    apiKeyField: "fal_api_key",
    endpoint: "https://fal.run/fal-ai/flux/dev",
    defaultModel: "fal-ai/flux/dev",
  },
  replicate: {
    name: "Replicate",
    category: "image",
    isFree: false,
    apiKeyField: "replicate_api_key",
    endpoint:
      "https://api.replicate.com/v1/models/black-forest-labs/flux-schnell/predictions",
    defaultModel: "black-forest-labs/flux-schnell",
  },
  pollinations: {
    name: "Pollinations.ai",
    category: "image",
    isFree: true,
    apiKeyField: null,
    endpoint: "https://image.pollinations.ai/prompt/{prompt}",
    defaultModel: "flux",
  },
  edge_tts: {
    name: "Edge TTS",
    category: "audio",
    isFree: true,
    apiKeyField: null,
    endpoint: "local:edge-tts",
    defaultModel: "en-US-AriaNeural",
  },
  elevenlabs: {
    name: "ElevenLabs",
    category: "audio",
    isFree: false,
    apiKeyField: "elevenlabs_api_key",
    endpoint: "https://api.elevenlabs.io/v1/text-to-speech",
    defaultModel: "eleven_multilingual_v2",
  },
};

function getPublicProviders(configuredFields = {}) {
  return Object.entries(providers).map(([id, provider]) => ({
    id,
    name: provider.name,
    category: provider.category,
    isFree: provider.isFree,
    apiKeyField: provider.apiKeyField,
    endpoint: provider.endpoint,
    defaultModel: provider.defaultModel,
    configured: provider.apiKeyField
      ? Boolean(configuredFields[provider.apiKeyField])
      : true,
  }));
}

module.exports = { providers, getPublicProviders };
