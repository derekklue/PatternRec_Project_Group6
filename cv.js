// cv.js
// Mapping classes
const PRODUCE_CLASSES   = ['Apple','Grapes','Peach','Raspberry'];
const VARIATION_CLASSES = {
  'Apple':    ['Whole','Sliced-Cored','In-Context'],
  'Grapes':   ['In a Bag','Loose Grapes','On the Vine'],
  'Peach':    ['Halved or Pitted','Sliced','Whole'],
  'Raspberry':['In a Container','Slightly Crushed','Small Group']
};

// ONNX run time
let produceSession, variationSessions = {};

export async function initModels() {
  produceSession = await ort.InferenceSession.create('models/produce.onnx');
  for (let f of PRODUCE_CLASSES) {
    variationSessions[f] =
      await ort.InferenceSession.create(`models/variation_${f}.onnx`);
  }
}

// Image to Tensor
function imgToTensor(img) {
  const c = document.createElement('canvas');
  c.width = c.height = 224;
  const ctx = c.getContext('2d');
  ctx.drawImage(img, 0, 0, 224, 224);
  const { data } = ctx.getImageData(0, 0, 224, 224);
  const arr = new Float32Array(1*3*224*224);
  for (let i=0; i<224*224; i++)
    for (let ch=0; ch<3; ch++)
      arr[ch*224*224 + i] = data[i*4 + ch]/255.0;
  return new ort.Tensor('float32', arr, [1,3,224,224]);
}


export async function classifyAll(imgEl) {
  const tensor = imgToTensor(imgEl);

  // Produce
  let { output: pOut } = await produceSession.run({ input: tensor });
  let pIdx = pOut.data.indexOf(Math.max(...pOut.data));
  let fruit = PRODUCE_CLASSES[pIdx];
  document.getElementById('produce-label').innerText = fruit;

  // Variation
  let { output: vOut } = await variationSessions[fruit].run({ input: tensor });
  let vIdx = vOut.data.indexOf(Math.max(...vOut.data));
  let variation = VARIATION_CLASSES[fruit][vIdx];
  document.getElementById('variation-label').innerText = variation;
}


export function wireUp() {
  const input = document.getElementById('img-upload');
  const img   = document.getElementById('preview');
  input.addEventListener('change', e => {
    img.src = URL.createObjectURL(e.target.files[0]);
    img.onload = () => classifyAll(img);
  });
}