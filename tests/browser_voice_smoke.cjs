/* Browser-component smoke test: verify a queued reply reaches speech synthesis. */
const fs = require("node:fs");
const vm = require("node:vm");
const assert = require("node:assert/strict");

const htmlPath = process.argv[2] || "components/browser_voice/index.html";
const html = fs.readFileSync(htmlPath, "utf8");
const match = html.match(/<script>([\s\S]*?)<\/script>/i);
assert.ok(match, "component must contain its JavaScript");

class FakeElement {
  constructor(id) {
    this.id = id;
    this.disabled = false;
    this.textContent = "";
    this.className = "";
    this.handlers = {};
    this.lastChild = { textContent: id === "status" ? "Voice link idle." : "" };
  }
  addEventListener(type, fn) { this.handlers[type] = fn; }
  click() { if (this.handlers.click) this.handlers.click(); }
}

class FakeUtterance {
  constructor(text) {
    this.text = String(text);
    this.lang = "";
    this.rate = 1;
    this.pitch = 1;
    this.voice = null;
    this.onstart = null;
    this.onend = null;
    this.onerror = null;
  }
}

const recognitions = [];
class FakeRecognition {
  constructor() {
    recognitions.push(this);
    this.lang = "en-IN";
    this.continuous = false;
    this.interimResults = false;
    this.maxAlternatives = 1;
    this.onstart = null;
    this.onend = null;
    this.onresult = null;
    this.onerror = null;
  }
  start() { if (this.onstart) this.onstart(); }
  stop() { if (this.onend) this.onend(); }
}

const elements = Object.fromEntries(
  ["start", "stop", "test", "replay", "speak", "status", "pulse", "heard"].map(id => [id, new FakeElement(id)])
);
const listeners = {};
const posted = [];
const utterances = [];
let nextTimerId = 0;
const timers = new Map();
function mockSetTimeout(callback, delay = 0) {
  const id = ++nextTimerId;
  timers.set(id, { callback, delay });
  return id;
}
function mockClearTimeout(id) { timers.delete(id); }
function runTimerByDelay(delay) {
  for (const [id, timer] of timers.entries()) {
    if (timer.delay === delay) {
      timers.delete(id);
      timer.callback();
      return true;
    }
  }
  return false;
}
const synth = {
  paused: false,
  speaking: false,
  silenceNext: false,
  cancel() { this.speaking = false; },
  resume() { this.paused = false; },
  getVoices() { return [{ lang: "en-IN", name: "Test voice" }]; },
  speak(utterance) {
    utterances.push(utterance);
    this.speaking = true;
    if (this.silenceNext) {
      this.silenceNext = false;
      return; // Simulate a browser that drops the first utterance.
    }
    if (utterance.onstart) utterance.onstart();
  }
};
const win = {
  speechSynthesis: synth,
  SpeechSynthesisUtterance: FakeUtterance,
  SpeechRecognition: FakeRecognition,
  webkitSpeechRecognition: FakeRecognition,
  parent: { postMessage(message) { posted.push(message); } },
  addEventListener(type, handler) { listeners[type] = handler; }
};
const context = {
  window: win,
  document: { getElementById(id) { return elements[id]; } },
  SpeechSynthesisUtterance: FakeUtterance,
  setTimeout: mockSetTimeout,
  clearTimeout: mockClearTimeout,
  Date,
  Math,
  String,
  Number,
  Boolean,
  Object
};
vm.runInNewContext(match[1], context, { timeout: 1000 });

function render(args) {
  assert.equal(typeof listeners.message, "function", "component should register its render listener");
  listeners.message({ data: { type: "streamlit:render", args } });
}
const baseArgs = {
  enabled: true,
  busy: false,
  language: "en-IN",
  rate: 0.92,
  speech_text: "",
  speech_sequence: 0,
  speech_kind: "reply",
  last_transcript: "",
  voice_error: "",
  answer_text: ""
};

render({ ...baseArgs });
elements.test.click();
assert.ok(utterances.some(u => u.text.includes("Smriti voice test")), "TEST VOICE click should call speechSynthesis");
const countAfterTest = utterances.length;

// The browser recognizer must report both readiness and a final transcript to Streamlit.
elements.start.click();
assert.equal(recognitions.length, 1, "Activate Listening should create a browser recognizer");
const recognition = recognitions[0];
assert.ok(posted.some(message =>
  message.type === "streamlit:setComponentValue"
  && message.value && message.value.event === "status"
  && message.value.code === "listening"
), "recognition startup should report microphone readiness");
recognition.onresult({
  resultIndex: 0,
  results: [{ 0: { transcript: "Hi Smriti, what can you do?" }, isFinal: true }]
});
assert.ok(posted.some(message =>
  message.type === "streamlit:setComponentValue"
  && message.value && message.value.event === "transcript"
  && message.value.transcript === "Hi Smriti, what can you do?"
), "final recognized questions must be sent to the Streamlit app");

render({
  ...baseArgs,
  speech_text: "A notebook is visible in the camera view.",
  answer_text: "You asked what I can do. I can describe the scene and help you recall saved notes.",
  speech_sequence: 1,
  speech_kind: "camera narration"
});
assert.equal(utterances.length, countAfterTest + 1, "new narration should be sent to speechSynthesis");
assert.equal(utterances.at(-1).text, "A notebook is visible in the camera view.");
assert.equal(elements.speak.disabled, false, "SPEAK ANSWER should enable once the app has an answer");
elements.speak.click();
assert.equal(
  utterances.at(-1).text,
  "You asked what I can do. I can describe the scene and help you recall saved notes.",
  "manual answer playback must speak latest JARVIS answer, not the most recent camera narration"
);

// Simulate a browser that fails to start the first AI utterance. The component
// should automatically retry once without requiring another user gesture.
synth.silenceNext = true;
render({ ...baseArgs, speech_text: "This answer should retry automatically.", speech_sequence: 2 });
assert.equal(utterances.at(-1).text, "This answer should retry automatically.");
assert.ok(runTimerByDelay(8000), "speech-start watchdog should allow a slow browser speech startup");
assert.ok(runTimerByDelay(300), "failed reply should schedule one automatic retry");
assert.equal(utterances.at(-1).text, "This answer should retry automatically.");
assert.equal(utterances.filter(u => u.text === "This answer should retry automatically.").length, 2);

// A reply that exists before the iframe's first render must not be marked consumed silently.
const secondElements = Object.fromEntries(
  ["start", "stop", "test", "replay", "speak", "status", "pulse", "heard"].map(id => [id, new FakeElement(id)])
);
const secondListeners = {};
const secondUtterances = [];
const secondSynth = {
  paused: false,
  cancel() {},
  resume() { this.paused = false; },
  getVoices() { return []; },
  speak(u) { secondUtterances.push(u); if (u.onstart) u.onstart(); }
};
const secondWindow = {
  speechSynthesis: secondSynth,
  SpeechSynthesisUtterance: FakeUtterance,
  SpeechRecognition: FakeRecognition,
  webkitSpeechRecognition: FakeRecognition,
  parent: { postMessage() {} },
  addEventListener(type, handler) { secondListeners[type] = handler; }
};
vm.runInNewContext(match[1], {
  window: secondWindow,
  document: { getElementById(id) { return secondElements[id]; } },
  SpeechSynthesisUtterance: FakeUtterance,
  setTimeout: mockSetTimeout,
  clearTimeout: mockClearTimeout,
  Date, Math, String, Number, Boolean, Object
}, { timeout: 1000 });
secondListeners.message({ data: { type: "streamlit:render", args: {
  ...baseArgs, speech_text: "A queued first reply.", speech_sequence: 7
} } });
assert.equal(secondUtterances.length, 1, "first-render queued speech must not be dropped");
assert.equal(secondUtterances[0].text, "A queued first reply.");

console.log("browser voice smoke test passed: test, recognition status and transcript, auto reply, manual speak answer, slow-start retry, and initial queued reply pass");
