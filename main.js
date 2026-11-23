
// ═══════════════════════════════════════════════════════════════════
// STUDENT SETTINGS
// ═══════════════════════════════════════════════════════════════════

const MODEL_PATH = 'assets/model.pict';

// FACE VISUALISATION MODE
// 'features'  = simplified eyebrows, eyes, nose and mouth
// 'keypoints' = all original FaceMesh keypoints

const FACE_RENDER_MODE = 'keypoints';

const SIZE = 256;
const OUTPUT_SIZE = 512;
const MAX_CANVAS = 512;
const MARGIN = 32;

// Face positioning
const FACE_WIDTH = 0.43;
const EYE_Y = 0.5;

// Generation interval
const INTERVAL = 1000;

// ═══════════════════════════════════════════════════════════════════
// VARIABLES
// ═══════════════════════════════════════════════════════════════════

let displaySize = MAX_CANVAS;

let inputCanvas;
let modelCanvas;
let outputCanvas;

let output;
let statusMsg;
let pix2pix;

let clearBtn;
let video;

let predictions = [];
let videoWidth = 0;
let videoHeight = 0;

let pix2pixReady = false;
let faceMeshReady = false;
let cameraReady = false;

let busy = false;
let lastRun = 0;

// ═══════════════════════════════════════════════════════════════════
// FACE FEATURES
// ═══════════════════════════════════════════════════════════════════

const FEATURES = [
  { parts: ['leftEyebrowUpper'], closed: false },
  { parts: ['rightEyebrowUpper'], closed: false },
  { parts: ['leftEyeUpper0', 'leftEyeLower0'], closed: true },
  { parts: ['rightEyeUpper0', 'rightEyeLower0'], closed: true },
  { parts: ['noseLeftCorner', 'noseBottom', 'noseRightCorner'], closed: false },
  { parts: ['lipsUpperInner', 'lipsLowerInner'], closed: true }
];

// ═══════════════════════════════════════════════════════════════════
// CANVAS SIZE
// ═══════════════════════════════════════════════════════════════════

function getCanvasSize() {
  return Math.floor(Math.min(MAX_CANVAS, windowWidth - MARGIN));
}

// ═══════════════════════════════════════════════════════════════════
// FLOYD-STEINBERG DITHER
// ═══════════════════════════════════════════════════════════════════

function ditherFloydSteinbergColor(pg) {
  pg.loadPixels();

  const w = pg.width;
  const h = pg.height;

  const r = new Float32Array(w * h);
  const g = new Float32Array(w * h);
  const b = new Float32Array(w * h);

  for (let i = 0; i < w * h; i++) {
    const o = i * 4;
    r[i] = pg.pixels[o];
    g[i] = pg.pixels[o + 1];
    b[i] = pg.pixels[o + 2];
  }

  const levels = 4;
  const step = 255 / (levels - 1);

  const clamp = v => Math.min(255, Math.max(0, v));
  const quantize = v => Math.round(Math.round(v / step) * step);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {

      const idx = y * w + x;

      const oldR = r[idx];
      const oldG = g[idx];
      const oldB = b[idx];

      const newR = quantize(oldR);
      const newG = quantize(oldG);
      const newB = quantize(oldB);

      r[idx] = newR;
      g[idx] = newG;
      b[idx] = newB;

      const errR = oldR - newR;
      const errG = oldG - newG;
      const errB = oldB - newB;

      if (x + 1 < w) {
        r[idx + 1] = clamp(r[idx + 1] + errR * 7 / 16);
        g[idx + 1] = clamp(g[idx + 1] + errG * 7 / 16);
        b[idx + 1] = clamp(b[idx + 1] + errB * 7 / 16);
      }

      if (y + 1 < h) {

        if (x - 1 >= 0) {
          r[idx + w - 1] = clamp(r[idx + w - 1] + errR * 3 / 16);
          g[idx + w - 1] = clamp(g[idx + w - 1] + errG * 3 / 16);
          b[idx + w - 1] = clamp(b[idx + w - 1] + errB * 3 / 16);
        }

        r[idx + w] = clamp(r[idx + w] + errR * 5 / 16);
        g[idx + w] = clamp(g[idx + w] + errG * 5 / 16);
        b[idx + w] = clamp(b[idx + w] + errB * 5 / 16);

        if (x + 1 < w) {
          r[idx + w + 1] = clamp(r[idx + w + 1] + errR * 1 / 16);
          g[idx + w + 1] = clamp(g[idx + w + 1] + errG * 1 / 16);
          b[idx + w + 1] = clamp(b[idx + w + 1] + errB * 1 / 16);
        }
      }
    }
  }

  for (let i = 0; i < w * h; i++) {
    const o = i * 4;

    pg.pixels[o] = r[i];
    pg.pixels[o + 1] = g[i];
    pg.pixels[o + 2] = b[i];
    pg.pixels[o + 3] = 255;
  }

  pg.updatePixels();
}

// ═══════════════════════════════════════════════════════════════════
// SETUP
// ═══════════════════════════════════════════════════════════════════

function setup() {

  pixelDensity(1);

  displaySize = getCanvasSize();

  inputCanvas = createCanvas(displaySize, displaySize);
  inputCanvas.class('border-box').parent('input');

  modelCanvas = createGraphics(SIZE, SIZE);
  modelCanvas.pixelDensity(1);

  outputCanvas = createGraphics(OUTPUT_SIZE, OUTPUT_SIZE);
  outputCanvas.pixelDensity(1);

  background(0);

  output = select('#output');
  statusMsg = select('#status');

  output.elt.style.maxWidth = '100%';
  output.elt.style.height = 'auto';

  statusMsg.html('Allow camera access to start face tracking...');

  // ── FACE TRACKER CONNECTION ──

  const faceTracker = select('#faceTracker');

  if (!faceTracker) {
    statusMsg.html('Error: faceTracker iframe not found.');
    return;
  }

  window.addEventListener('message', function(event) {

    if (event.origin !== window.location.origin) return;

    if (event.source !== faceTracker.elt.contentWindow) return;

    if (!event.data) return;

    if (event.data.type === 'face-tracker-ready') {

      cameraReady = true;
      faceMeshReady = true;

      updateStatus();

    } else if (event.data.type === 'face-tracker-predictions') {

      predictions = event.data.predictions || [];

      videoWidth = event.data.width;
      videoHeight = event.data.height;

    } else if (event.data.type === 'face-tracker-error') {

      statusMsg.html('Face tracking failed: ' + event.data.message);

    }

  });

  // ── CLEAR BUTTON ──

  clearBtn = select('#clearBtn');

  if (clearBtn) {
    clearBtn.mousePressed(clearCanvas);
  }

  describe(
    'A 512 by 512 canvas showing white facial landmark outlines on a black background, tracked from the camera.'
  );

  // ── LOAD PIX2PIX ──

  pix2pix = ml5.pix2pix(MODEL_PATH, modelLoaded);
}

// ═══════════════════════════════════════════════════════════════════
// DRAW LOOP
// ═══════════════════════════════════════════════════════════════════

function draw() {

  drawLandmarks();

  if (
    pix2pixReady &&
    faceMeshReady &&
    cameraReady &&
    !busy &&
    millis() - lastRun >= INTERVAL
  ) {

    lastRun = millis();

    runTransferCycle();

  }
}

// ═══════════════════════════════════════════════════════════════════
// AUTOMATIC GENERATION
// ═══════════════════════════════════════════════════════════════════

async function runTransferCycle() {

  busy = true;

  try {

    if (predictions.length === 0) {

      statusMsg.html('No face detected. Center your face in the camera.');

      return;
    }

    statusMsg.html('Generating face...');

    await transfer();

  } catch (error) {

    console.error('Face generation failed:', error);

    statusMsg.html('Generation failed: ' + (error.message || error));

  } finally {

    busy = false;

  }
}

// ═══════════════════════════════════════════════════════════════════
// DRAW FACE LANDMARKS
// ═══════════════════════════════════════════════════════════════════


function drawLandmarks() {

  background(0);

  if (predictions.length === 0) return;

  const face = predictions[0];

  // Choose the visualisation mode
  if (FACE_RENDER_MODE === 'keypoints') {
    drawOriginalKeypoints(face);
  } else {
    drawFeatureLandmarks(face);
  }
}


// ═══════════════════════════════════════════════════════════════════
// ORIGINAL FACEMESH KEYPOINTS
// ═══════════════════════════════════════════════════════════════════

function drawOriginalKeypoints(face) {

  // Legacy ml5 FaceMesh uses scaledMesh.
  // Newer versions may provide keypoints as objects.

  let points = [];

  if (face.scaledMesh) {
    points = face.scaledMesh.map(p => ({
      x: p[0],
      y: p[1]
    }));
  } else if (face.keypoints) {
    points = face.keypoints.map(p => ({
      x: p.x,
      y: p.y
    }));
  }

  if (points.length === 0) return;

  // Find the face boundaries
  const minX = Math.min(...points.map(p => p.x));
  const maxX = Math.max(...points.map(p => p.x));
  const minY = Math.min(...points.map(p => p.y));
  const maxY = Math.max(...points.map(p => p.y));

  const faceWidth = maxX - minX;
  const faceHeight = maxY - minY;

  if (faceWidth <= 0 || faceHeight <= 0) return;

  // Fit the face inside the square canvas
  const availableSize = displaySize * 0.75;

  const scale = Math.min(
    availableSize / faceWidth,
    availableSize / faceHeight
  );

  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;

  const toX = x =>
    displaySize / 2 + (x - centerX) * scale;

  const toY = y =>
    displaySize / 2 + (y - centerY) * scale;

  // Plot every original FaceMesh keypoint
  noStroke();
  fill(255);

  for (const point of points) {

    if (!Number.isFinite(point.x) ||
        !Number.isFinite(point.y)) continue;

    circle(
      toX(point.x),
      toY(point.y),
      3
    );

  }
}


// ═══════════════════════════════════════════════════════════════════
// SIMPLIFIED FACIAL FEATURES (YOUR ORIGINAL MODE)
// ═══════════════════════════════════════════════════════════════════

function drawFeatureLandmarks(face) {

  const a = face.annotations;

  if (!a || videoWidth === 0 || videoHeight === 0) return;

  stroke(255);
  strokeWeight(1);
  strokeCap(ROUND);
  strokeJoin(ROUND);
  noFill();

  const leftBrow = (a.leftEyebrowUpper || []).slice(0, -2);
  const rightBrow = (a.rightEyebrowUpper || []).slice(0, -2);

  const isPoint = p =>
    p && Number.isFinite(p[0]) && Number.isFinite(p[1]);

  const browPoints = leftBrow.concat(rightBrow).filter(isPoint);

  const eyePoints = (a.leftEyeUpper0 || [])
    .concat(a.rightEyeUpper0 || [])
    .filter(isPoint);

  if (browPoints.length < 2 || eyePoints.length === 0) return;

  const minX = Math.min(...browPoints.map(p => p[0]));
  const maxX = Math.max(...browPoints.map(p => p[0]));

  const eyeY = eyePoints.reduce(
    (sum, p) => sum + p[1], 0
  ) / eyePoints.length;

  const browWidth = maxX - minX;

  if (browWidth <= 0) return;

  const scale = (FACE_WIDTH * displaySize) / browWidth;
  const midX = (minX + maxX) / 2;

  const toX = x =>
    displaySize / 2 - (x - midX) * scale;

  const toY = y =>
    EYE_Y * displaySize + (y - eyeY) * scale;

  FEATURES.forEach(function(feature) {

    const pts = [];

    feature.parts.forEach(function(name, i) {

      if (!a[name]) return;

      let group = [...a[name]];

      if (i % 2) group.reverse();

      pts.push(...group.filter(isPoint));

    });

    if (pts.length < 2) return;

    beginShape();

    pts.forEach(p => {
      vertex(toX(p[0]), toY(p[1]));
    });

    if (feature.closed) {
      endShape(CLOSE);
    } else {
      endShape();
    }

  });

}


// ═══════════════════════════════════════════════════════════════════
// MODEL LOADING
// ═══════════════════════════════════════════════════════════════════

function modelLoaded() {

  pix2pixReady = true;

  updateStatus();

}

// ═══════════════════════════════════════════════════════════════════
// STATUS
// ═══════════════════════════════════════════════════════════════════

function updateStatus() {

  if (!cameraReady) {

    statusMsg.html('Allow camera access to start face tracking...');

  } else if (pix2pixReady && faceMeshReady) {

    statusMsg.html('Face tracking ready. Center your face in the camera.');

  } else if (pix2pixReady) {

    statusMsg.html('Pix2Pix loaded. Waiting for face tracker...');

  } else if (faceMeshReady) {

    statusMsg.html('Face tracker loaded. Loading Pix2Pix...');

  }

}

// ═══════════════════════════════════════════════════════════════════
// CLEAR CANVAS
// ═══════════════════════════════════════════════════════════════════

function clearCanvas() {

  predictions = [];

  background(0);

  output.elt.src = '';
  output.elt.alt = 'The AI-generated face output will appear here after tracking a face.';

  updateStatus();

}

// ═══════════════════════════════════════════════════════════════════
// PIX2PIX TRANSFER
// ═══════════════════════════════════════════════════════════════════

async function transfer() {

  // Copy tracked face drawing into the model canvas

  modelCanvas.background(0);

  modelCanvas.image(
    get(),
    0,
    0,
    SIZE,
    SIZE
  );

  // Run Pix2Pix

  const result = await new Promise((resolve, reject) => {

    pix2pix.transfer(modelCanvas.elt, function(error, transferResult) {

      if (error) {
        reject(error);
      } else {
        resolve(transferResult);
      }

    });

  });

  if (!result || !result.src) {
    throw new Error('The model returned no image.');
  }

  // Load generated image

  const generatedImage = await new Promise((resolve, reject) => {

    loadImage(result.src, resolve, reject);

  });

  // Resize output

  outputCanvas.clear();

  outputCanvas.image(
    generatedImage,
    0,
    0,
    OUTPUT_SIZE,
    OUTPUT_SIZE
  );

  // Apply dithering

  ditherFloydSteinbergColor(outputCanvas);

  // Convert to image

  const blob = await new Promise((resolve, reject) => {

    outputCanvas.canvas.toBlob(resultBlob => {

      if (resultBlob) {
        resolve(resultBlob);
      } else {
        reject(new Error('Could not create generated image.'));
      }

    });

  });

  // Display output

  const old = output.elt.src;

  output.elt.src = URL.createObjectURL(blob);

  if (old.startsWith('blob:')) {
    URL.revokeObjectURL(old);
  }

  output.elt.alt =
    'AI-generated face produced from tracked facial landmarks.';

  statusMsg.html('Face tracking active. Generating live.');

}
