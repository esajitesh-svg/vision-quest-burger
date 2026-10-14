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

class FakeRecognition {
  constructor() {
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
  ["start", "stop", "test", "replay", "status", "pulse", "heard"].map(id => [id, new FakeElement(id)])
);
const listeners = {};
const posted = [];
const utterances = [];
const synth = {
  paused: false,
  speaking: false,
  cancel() { this.speaking = false; },
  resume() { this.paused = false; },
  getVoices() { return [{ lang: "en-IN", name: "Test voice" }]; },
  speak(utterance) {
    utterances.push(utterance);
    this.speaking = true;
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
  setTimeout() { return 1; },
  clearTimeout() {},
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
  voice_error: ""
};

render({ ...baseArgs });
elements.test.click();
assert.ok(utterances.some(u => u.text.includes("Smriti voice test")), "TEST VOICE click should call speechSynthesis");
const countAfterTest = utterances.length;

render({ ...baseArgs, speech_text: "The camera sees a blue notebook.", speech_sequence: 1 });
assert.equal(utterances.length, countAfterTest + 1, "new AI reply should be sent to speechSynthesis");
assert.equal(utterances.at(-1).text, "The camera sees a blue notebook.");

// A reply that exists before the iframe's first render must not be marked consumed silently.
const secondElements = Object.fromEntries(
  ["start", "stop", "test", "replay", "status", "pulse", "heard"].map(id => [id, new FakeElement(id)])
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
  setTimeout() { return 1; },
  clearTimeout() {},
  Date, Math, String, Number, Boolean, Object
}, { timeout: 1000 });
secondListeners.message({ data: { type: "streamlit:render", args: {
  ...baseArgs, speech_text: "A queued first reply.", speech_sequence: 7
} } });
assert.equal(secondUtterances.length, 1, "first-render queued speech must not be dropped");
assert.equal(secondUtterances[0].text, "A queued first reply.");

console.log("browser voice smoke test passed: test, new reply, and initial queued reply all reach speechSynthesis");
