const fs = require("fs");
const fsPromises = require("fs/promises");
const path = require("path");
const { execFile } = require("child_process");
const { randomBytes } = require("crypto");
const { once } = require("events");
const { finished } = require("stream/promises");
const ffmpeg = require("fluent-ffmpeg");

const outputsDirectory = path.join(__dirname, "outputs");

async function ensureOutputsDirectory() {
  await fsPromises.mkdir(outputsDirectory, { recursive: true });
}

async function ensurePlaceholderVideoImage() {
  await ensureOutputsDirectory();
  const filePath = path.join(outputsDirectory, "placeholder-video.ppm");
  const existing = await fsPromises.stat(filePath).catch(() => null);
  if (existing?.size > 0) return filePath;

  const width = 1024;
  const height = 1024;
  const pixels = Buffer.alloc(width * height * 3);
  for (let y = 0; y < height; y += 1) {
    const blend = y / (height - 1);
    const red = Math.round(23 + (112 - 23) * blend);
    const green = Math.round(21 + (87 - 21) * blend);
    const blue = Math.round(47 + (233 - 47) * blend);
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 3;
      pixels[offset] = red;
      pixels[offset + 1] = green;
      pixels[offset + 2] = blue;
    }
  }

  await fsPromises.writeFile(
    filePath,
    Buffer.concat([Buffer.from(`P6\n${width} ${height}\n255\n`), pixels])
  );
  return filePath;
}

function safeExecutionId(executionId) {
  return String(executionId).replace(/[^a-zA-Z0-9_-]/g, "");
}

function outputName(executionId, step, attemptId, extension) {
  const suffix = attemptId ? `-${String(attemptId).replace(/[^a-f0-9]/gi, "").slice(0, 12)}` : "";
  return `${safeExecutionId(executionId)}-step-${step}${suffix}.${extension}`;
}

function outputUrl(filename) {
  return `/api/outputs/${encodeURIComponent(filename)}`;
}

function imageExtension(contentType) {
  if (contentType.includes("png")) return "png";
  if (contentType.includes("webp")) return "webp";
  return "jpg";
}

function runCommand(command, args, timeoutMs, signal) {
  return new Promise((resolve, reject) => {
    execFile(
      command,
      args,
      { timeout: timeoutMs, signal, maxBuffer: 1024 * 1024 },
      (error, stdout, stderr) => {
        if (error) {
          const detail = String(stderr || error.message).trim();
          reject(new Error(detail || `${command} failed.`));
          return;
        }
        resolve({ stdout, stderr });
      }
    );
  });
}

async function synthesizeSpeech({
  text,
  executionId,
  attemptId,
  stepNumber = 9,
  voice = "en-US-AriaNeural",
  signal,
}) {
  const script = String(text || "").trim();
  if (!script) {
    throw new Error(`Step ${stepNumber} could not find the Step 4 script text.`);
  }

  await ensureOutputsDirectory();
  const filename = outputName(executionId, String(stepNumber).padStart(2, "0"), attemptId, "mp3");
  const filePath = path.join(outputsDirectory, filename);

  try {
    await runCommand(
      "edge-tts",
      ["--voice", voice, "--text", script, "--write-media", filePath],
      120000,
      signal
    );
  } catch (error) {
    throw new Error(`Edge TTS failed: ${error.message}`);
  }

  const stats = await fsPromises.stat(filePath).catch(() => null);
  if (!stats || stats.size === 0) {
    throw new Error("Edge TTS completed but did not create an audio file.");
  }

  return { filePath, url: outputUrl(filename), voice };
}

async function synthesizeElevenLabs({ text, apiKey, executionId, attemptId, signal }) {
  if (!apiKey) throw new Error("Add an ElevenLabs API key in Settings.");
  const script = String(text || "").trim();
  if (!script) throw new Error("Step 10 could not find the Step 4 script text.");
  const voicesResponse = await fetch("https://api.elevenlabs.io/v1/voices", {
    headers: { "xi-api-key": apiKey },
    signal,
  });
  if (!voicesResponse.ok) throw new Error(`ElevenLabs voices returned HTTP ${voicesResponse.status}.`);
  const voices = (await voicesResponse.json()).voices || [];
  const selectedVoice = voices.find((voice) => voice.name === "Rachel") || voices[0];
  if (!selectedVoice?.voice_id) throw new Error("ElevenLabs did not return an available voice.");
  const response = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${encodeURIComponent(selectedVoice.voice_id)}`, {
    method: "POST",
    headers: { "xi-api-key": apiKey, "Content-Type": "application/json", Accept: "audio/mpeg" },
    body: JSON.stringify({ text: script, model_id: "eleven_multilingual_v2" }),
    signal,
  });
  if (!response.ok) {
    const body = await response.text();
    let message = body.slice(0, 250);
    try {
      const parsed = JSON.parse(body);
      message = parsed.detail?.message || parsed.detail?.status || message;
    } catch {}
    throw new Error(`ElevenLabs returned HTTP ${response.status}: ${message}`);
  }
  const chunks = [];
  let size = 0;
  for await (const chunk of response.body) {
    if (signal?.aborted) throw new Error("Voiceover was cancelled.");
    size += chunk.length;
    if (size > 25 * 1024 * 1024) throw new Error("ElevenLabs audio exceeded 25 MB.");
    chunks.push(chunk);
  }
  if (!size || signal?.aborted) throw new Error("ElevenLabs returned no audio.");
  await ensureOutputsDirectory();
  const filename = outputName(executionId, "10", attemptId, "mp3");
  const filePath = path.join(outputsDirectory, filename);
  await fsPromises.writeFile(filePath, Buffer.concat(chunks, size));
  return { filePath, url: outputUrl(filename), voice: selectedVoice.name || "ElevenLabs", provider: "elevenlabs" };
}

async function downloadImage(imageUrl, executionId, signal) {
  if (imageUrl === "/placeholder-image.svg") {
    return {
      filePath: await ensurePlaceholderVideoImage(),
      temporary: false,
    };
  }

  if (/^\/api\/outputs\//.test(String(imageUrl || ""))) {
    const filePath = localPathFromOutputUrl(imageUrl);
    const expectedPrefix = `${safeExecutionId(executionId)}-step-07`;
    if (!filePath || !path.basename(filePath).startsWith(expectedPrefix)) {
      throw new Error("Step 13 received an invalid saved Step 7 image path.");
    }
    const stats = await fsPromises.stat(filePath).catch(() => null);
    if (!stats || stats.size === 0) {
      throw new Error("The saved Step 7 image file is missing.");
    }
    return { filePath, temporary: false };
  }

  if (!/^https:\/\//i.test(String(imageUrl || ""))) {
    throw new Error("Step 13 could not find a valid Step 7 image URL.");
  }

  const parsedUrl = new URL(imageUrl);
  const allowedHosts = [
    "image.pollinations.ai",
    "fal.media",
    "replicate.delivery",
  ];
  const allowed = allowedHosts.some(
    (host) =>
      parsedUrl.hostname === host || parsedUrl.hostname.endsWith(`.${host}`)
  );
  if (!allowed) {
    throw new Error(
      `The Step 7 image host "${parsedUrl.hostname}" is not an approved provider host.`
    );
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120000);
  let response;

  try {
    response = await fetch(imageUrl, {
      signal: signal ? AbortSignal.any([controller.signal, signal]) : controller.signal,
    });
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error("The Step 7 image download timed out.");
    }
    throw new Error(`The Step 7 image could not be downloaded: ${error.message}`);
  } finally {
    clearTimeout(timeout);
  }

  if (!response.ok) {
    throw new Error(
      `The Step 7 image download returned HTTP ${response.status}.`
    );
  }

  const contentType = response.headers.get("content-type") || "";
  if (!contentType.startsWith("image/")) {
    throw new Error("The Step 7 URL did not return an image.");
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("The Step 7 image response had no readable body.");
  }

  const chunks = [];
  let totalBytes = 0;
  while (true) {
    if (signal?.aborted) throw new Error("Image download was cancelled.");
    const { done, value } = await reader.read();
    if (done) break;
    totalBytes += value.byteLength;
    if (totalBytes > 25 * 1024 * 1024) {
      await reader.cancel();
      throw new Error("The Step 7 image is larger than the 25 MB limit.");
    }
    chunks.push(Buffer.from(value));
  }

  const bytes = Buffer.concat(chunks, totalBytes);
  if (!bytes.length) {
    throw new Error("The Step 7 image download was empty.");
  }

  const extension = imageExtension(contentType);
  const filePath = path.join(
    outputsDirectory,
    `${safeExecutionId(executionId)}-step-07-source-${randomBytes(6).toString("hex")}.${extension}`
  );
  await fsPromises.writeFile(filePath, bytes);
  return { filePath, temporary: true };
}

async function persistImage({
  imageUrl,
  imageBuffer,
  contentType = "image/jpeg",
  executionId,
  attemptId,
  signal,
}) {
  if (signal?.aborted) throw new Error("Image generation was cancelled.");
  if (imageUrl === "/placeholder-image.svg") {
    return { filePath: path.join(__dirname, "public", "placeholder-image.svg"), url: imageUrl };
  }

  await ensureOutputsDirectory();
  let extension = imageExtension(contentType);
  let downloaded;

  if (Buffer.isBuffer(imageBuffer) && imageBuffer.length > 0) {
    if (imageBuffer.length > 25 * 1024 * 1024) {
      throw new Error("The generated Step 7 image is larger than 25 MB.");
    }
  } else {
    downloaded = await downloadImage(imageUrl, executionId, signal);
    extension = path.extname(downloaded.filePath).slice(1) || extension;
  }

  const filename = outputName(executionId, "07", attemptId, extension);
  const filePath = path.join(outputsDirectory, filename);
  if (signal?.aborted) throw new Error("Image generation was cancelled.");
  if (Buffer.isBuffer(imageBuffer) && imageBuffer.length > 0) {
    await fsPromises.writeFile(filePath, imageBuffer);
  } else {
    try {
      await fsPromises.copyFile(downloaded.filePath, filePath);
    } finally {
      if (downloaded.temporary) {
        await fsPromises.unlink(downloaded.filePath).catch(() => {});
      }
    }
  }

  const stats = await fsPromises.stat(filePath).catch(() => null);
  if (!stats || stats.size === 0) {
    throw new Error("Step 7 completed but did not save an image file.");
  }
  return { filePath, url: outputUrl(filename) };
}

async function probeDuration(filePath, signal) {
  const { stdout } = await runCommand(
    "ffprobe",
    ["-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path.resolve(filePath)],
    15000,
    signal
  );
  const duration = Number(stdout.trim());
  if (!Number.isFinite(duration) || duration <= 0) {
    throw new Error("The voiceover has no valid audio duration.");
  }
  return duration;
}

function renderKenBurns({ imagePath, outputPath, signal }) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new Error("Image-to-video was cancelled."));
    const command = ffmpeg()
      .input(path.resolve(imagePath))
      .inputOptions(["-loop 1"])
      .videoFilters([
        "scale=800:1422:force_original_aspect_ratio=increase",
        "crop=800:1422",
        "zoompan=z='min(zoom+0.0008,1.12)':d=1:s=720x1280:fps=30",
      ])
      .outputOptions(["-nostdin", "-y", "-frames:v 150", "-c:v libx264", "-preset veryfast", "-pix_fmt yuv420p", "-movflags +faststart"])
      .output(path.resolve(outputPath));
    let settled = false;
    const finish = (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      signal?.removeEventListener("abort", abort);
      error ? reject(error) : resolve();
    };
    const abort = () => {
      command.kill("SIGKILL");
      finish(new Error("Image-to-video was cancelled."));
    };
    const timeout = setTimeout(() => {
      command.kill("SIGKILL");
      finish(new Error("Image-to-video timed out."));
    }, 175000);
    signal?.addEventListener("abort", abort, { once: true });
    command
      .on("end", () => finish())
      .on("error", (error, stdout, stderr) => finish(new Error(`Ken Burns FFmpeg failed: ${String(stderr || error.message).slice(-1500)}`)))
      .run();
  });
}

async function createKenBurnsClip({ imageUrl, executionId, attemptId, signal }) {
  await ensureOutputsDirectory();
  const image = await downloadImage(imageUrl, executionId, signal);
  try {
    const filename = outputName(executionId, "08", attemptId, "mp4");
    const filePath = path.join(outputsDirectory, filename);
    await renderKenBurns({ imagePath: image.filePath, outputPath: filePath, signal });
    const stat = await fsPromises.stat(filePath);
    if (!stat.size) throw new Error("Step 8 generated an empty video.");
    return { filePath, url: outputUrl(filename), duration: 5 };
  } finally {
    if (image.temporary) await fsPromises.unlink(image.filePath).catch(() => {});
  }
}

async function localImageDataUri(imageUrl, executionId) {
  const image = await downloadImage(imageUrl, executionId);
  try {
    const bytes = await fsPromises.readFile(image.filePath);
    if (!bytes.length || bytes.length > 25 * 1024 * 1024) throw new Error("Step 7 image is empty or too large for Fal.ai.");
    const extension = path.extname(image.filePath).toLowerCase();
    if (![".jpg", ".jpeg", ".png", ".webp"].includes(extension)) {
      throw new Error("Fal.ai requires a JPG, PNG or WebP image from Step 7.");
    }
    const type = extension === ".png" ? "image/png" : extension === ".webp" ? "image/webp" : "image/jpeg";
    return `data:${type};base64,${bytes.toString("base64")}`;
  } finally {
    if (image.temporary) await fsPromises.unlink(image.filePath).catch(() => {});
  }
}

async function persistFalVideo({ videoUrl, executionId, attemptId, signal }) {
  const url = new URL(videoUrl);
  if (url.protocol !== "https:" || !(
    url.hostname === "fal.media" ||
    url.hostname.endsWith(".fal.media") ||
    (url.hostname === "storage.googleapis.com" && url.pathname.startsWith("/falserverless/"))
  )) throw new Error("Fal.ai returned an unsupported video host.");
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`Fal.ai video download returned HTTP ${response.status}.`);
  await ensureOutputsDirectory();
  const filename = outputName(executionId, "08", attemptId, "mp4");
  const filePath = path.join(outputsDirectory, filename);
  const stream = fs.createWriteStream(filePath);
  let size = 0;
  try {
    for await (const chunk of response.body) {
      if (signal?.aborted) throw new Error("Video download was cancelled.");
      size += chunk.length;
      if (size > 150 * 1024 * 1024) throw new Error("Fal.ai video exceeded 150 MB.");
      if (!stream.write(chunk)) await once(stream, "drain");
    }
    stream.end();
    await finished(stream);
    if (!size || signal?.aborted) throw new Error("Fal.ai returned an empty video.");
    return { filePath, url: outputUrl(filename), duration: 5 };
  } catch (error) {
    stream.destroy();
    await fsPromises.unlink(filePath).catch(() => {});
    throw error;
  }
}

async function resizeAssemblyImage({ imagePath, outputPath, signal }) {
  await runCommand(
    "ffmpeg",
    [
      "-nostdin", "-y", "-i", path.resolve(imagePath),
      "-vf", "scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2",
      "-frames:v", "1", path.resolve(outputPath),
    ],
    30000,
    signal
  );
}

function renderVideo({ imagePath, audioPath, outputPath, duration, signal, log }) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new Error("Video assembly was cancelled."));
      return;
    }
    const command = ffmpeg()
      .input(path.resolve(imagePath))
      .inputOptions(["-loop 1", "-framerate 24"])
      .input(path.resolve(audioPath))
      .outputOptions([
        "-nostdin",
        "-y",
        "-c:v libx264",
        "-preset ultrafast",
        "-tune stillimage",
        "-pix_fmt yuv420p",
        "-c:a aac",
        "-shortest",
        "-movflags +faststart",
      ])
      .duration(duration)
      .output(path.resolve(outputPath));

    let settled = false;
    let nextMilestone = 25;
    const finish = (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      signal?.removeEventListener("abort", abort);
      error ? reject(error) : resolve();
    };
    const abort = () => {
      command.kill("SIGKILL");
      finish(new Error("Video assembly was cancelled."));
    };
    const timeout = setTimeout(() => {
      command.kill("SIGKILL");
      finish(new Error("Video assembly timed out."));
    }, 300000);
    signal?.addEventListener("abort", abort, { once: true });
    command
      .on("start", () => log("start: encoding image and audio"))
      .on("progress", (progress) => {
        const seconds = String(progress.timemark || "0:0:0").split(":").reduce(
          (total, part) => total * 60 + Number(part), 0
        );
        const elapsed = Number.isFinite(seconds) ? seconds : 0;
        const percent = Math.min(100, Math.max(
          (elapsed / duration) * 100,
          (Number(progress.frames) / (duration * 24)) * 100 || 0
        ));
        while (nextMilestone <= 75 && percent >= nextMilestone) {
          log(`${nextMilestone}%`);
          nextMilestone += 25;
        }
      })
      .on("end", () => finish())
      .on("error", (error, stdout, stderr) => {
        const detail = String(stderr || error.message).trim().slice(-2000);
        finish(new Error(`FFmpeg video assembly failed: ${detail}`));
      })
      .run();
  });
}

async function assembleVideo({
  imageUrl,
  audioPath,
  executionId,
  attemptId,
  signal,
  attempt = 1,
}) {
  const log = (message) => console.info(`[Step 13 ${executionId} attempt ${attempt}] ${message}`);
  await ensureOutputsDirectory();
  const audioStats = await fsPromises.stat(audioPath).catch(() => null);
  if (!audioStats || audioStats.size === 0) {
    throw new Error("Step 13 could not find the Step 10 audio file.");
  }

  log("start: loading Step 7 image and Step 10 audio");
  const image = await downloadImage(imageUrl, executionId, signal);
  const resizedPath = path.join(outputsDirectory, outputName(executionId, "13", attemptId, "jpg"));
  try {
    const duration = await probeDuration(audioPath, signal);
    if (signal?.aborted) throw new Error("Video assembly was cancelled.");
    log("resizing Step 7 image to 720x1280");
    await resizeAssemblyImage({ imagePath: image.filePath, outputPath: resizedPath, signal });
    if (signal?.aborted) throw new Error("Video assembly was cancelled.");
    const filename = outputName(executionId, "13", attemptId, "mp4");
    const filePath = path.join(outputsDirectory, filename);

    await renderVideo({
      imagePath: resizedPath,
      audioPath,
      outputPath: filePath,
      duration,
      signal,
      log,
    });

    const stats = await fsPromises.stat(filePath).catch(() => null);
    if (!stats || stats.size === 0) {
      throw new Error("FFmpeg completed but did not create a video file.");
    }

    log("done");
    return { filePath, url: outputUrl(filename), duration };
  } finally {
    await fsPromises.unlink(resizedPath).catch(() => {});
    if (image.temporary) {
      await fsPromises.unlink(image.filePath).catch(() => {});
    }
  }
}

function localPathFromOutputUrl(url) {
  const match = String(url || "").match(/^\/api\/outputs\/([^/?#]+)$/);
  if (!match) return "";
  let filename;
  try {
    filename = path.basename(decodeURIComponent(match[1]));
  } catch {
    return "";
  }
  return path.join(outputsDirectory, filename);
}

module.exports = {
  assembleVideo,
  createKenBurnsClip,
  ensureOutputsDirectory,
  localImageDataUri,
  localPathFromOutputUrl,
  outputsDirectory,
  persistFalVideo,
  persistImage,
  synthesizeElevenLabs,
  synthesizeSpeech,
};
                                 
