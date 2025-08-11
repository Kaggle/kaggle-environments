import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.118/build/three.module.js';

import {OrbitControls} from 'https://cdn.jsdelivr.net/npm/three@0.118/examples/jsm/controls/OrbitControls.js';
import {FBXLoader} from "https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/loaders/FBXLoader.js";
import { SkeletonUtils } from 'https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/utils/SkeletonUtils.js';
import { CSS2DRenderer, CSS2DObject } from 'https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/renderers/CSS2DRenderer.js';


class BasicWorldDemo {
  constructor() {
    this._Initialize();
  }

  _Initialize() {
    // WebGL Renderer
    this._threejs = new THREE.WebGLRenderer({
      antialias: true,
    });
    this._threejs.shadowMap.enabled = true;
    this._threejs.shadowMap.type = THREE.PCFSoftShadowMap;
    this._threejs.setPixelRatio(window.devicePixelRatio);
    this._threejs.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(this._threejs.domElement);

    // CSS2D Renderer
    this._labelRenderer = new CSS2DRenderer();
    this._labelRenderer.setSize(window.innerWidth, window.innerHeight);
    this._labelRenderer.domElement.style.position = 'absolute';
    this._labelRenderer.domElement.style.top = '0px';
    this._labelRenderer.domElement.style.pointerEvents = 'none';
    document.body.appendChild(this._labelRenderer.domElement);

    window.addEventListener('resize', () => {
      this._OnWindowResize();
    }, false);

    const fov = 60;
    const aspect = window.innerWidth / window.innerHeight;
    const near = 1.0;
    const far = 100000.0;
    this._camera = new THREE.PerspectiveCamera(fov, aspect, near, far);
    this._camera.position.set(0, 0, 50);

    this._scene = new THREE.Scene();
    
    this._createSkybox();

    const hemisphereLight = new THREE.HemisphereLight(0xffffbb, 0x080820, 1.5);
    this._scene.add(hemisphereLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 2.0);
    dirLight.position.set(20, 50, 10);
    dirLight.castShadow = true;
    this._scene.add(dirLight);

    this._controls = new OrbitControls(this._camera, this._threejs.domElement);
    this._controls.target.set(0, 0, 0);
    this._controls.update();

    this._LoadModels();
    this._RAF();
  }

  _createSkybox() {
    const skyboxSize = 1000;
    const skyboxGeo = new THREE.BoxGeometry(skyboxSize, skyboxSize, skyboxSize);
    const black = 0x000000;

    const backPanelMaterial = new THREE.MeshBasicMaterial({ side: THREE.BackSide });

    // The order is: right, left, top, bottom, front, back
    // The "front" face (+z) is the one behind the characters, serving as the backdrop.
    const materials = [
        new THREE.MeshBasicMaterial({ color: black, side: THREE.BackSide }), // right
        new THREE.MeshBasicMaterial({ color: black, side: THREE.BackSide }), // left
        new THREE.MeshBasicMaterial({ color: black, side: THREE.BackSide }), // top
        new THREE.MeshBasicMaterial({ color: black, side: THREE.BackSide }), // bottom
        backPanelMaterial,                                                    // front (the backdrop)
        new THREE.MeshBasicMaterial({ color: black, side: THREE.BackSide })   // back (behind camera)
    ];

    const skybox = new THREE.Mesh(skyboxGeo, materials);
    this._scene.add(skybox);

    const backCanvas = document.createElement('canvas');
    const backContext = backCanvas.getContext('2d');

    const canvasSize = 2048
    backCanvas.width = canvasSize;
    backCanvas.height = canvasSize;
    
    const moonImage = new Image();
    moonImage.onload = () => {
        backContext.fillStyle = 'black';
        backContext.fillRect(0, 0, backCanvas.width, backCanvas.height);
        const moonSize = 500;
        backContext.drawImage(moonImage, moonSize, moonSize, moonSize, moonSize);
//        backContext.drawImage(moonImage, backCanvas.width - moonSize - 50, 50, moonSize, moonSize);
        
        const backTexture = new THREE.CanvasTexture(backCanvas);
        
        backPanelMaterial.map = backTexture;
        backPanelMaterial.needsUpdate = true;
    };
    moonImage.onerror = () => {
        console.error("Failed to load moon texture for skybox.");
    };
    moonImage.src = 'assets/moon4.png';
  }

  _LoadModels() {
    const fbxLoader = new FBXLoader();
    fbxLoader.load('assets/Idle_stickman.fbx', (fbx) => {
        const stickmanModel = this._NormalizeModel(fbx);
        const stickmanGroup = new THREE.Group();
        const idleClip = fbx.animations[0];

        const names = ["gemini-2.5-pro", "gpt-4.1", "claude-4-sonnet", "grok4", "deepseek-r1", "kimi-k2", "qwen3", "gemini-2.5-flash"];
        const brands = ["gemini", "chatgpt", "claude", "grok", "deepseek", "kimi", "qwen", "gemini"];
        const URLS = {
            "gemini": "assets/logo/gemini.png",
            "chatgpt": "assets/logo/chatgpt.png",
            "claude": "assets/logo/claude.png",
            "grok": "assets/logo/grok.png",
            "deepseek": "assets/logo/deepseek.png",
            "kimi": "assets/logo/kimi.png",
            "qwen": "assets/logo/qwen.png"
        };
        
        const numStickmen = 8;
        const radius = 15;
        const sectorAngleDegrees = 60;
        const sectorAngleRadians = sectorAngleDegrees * (Math.PI / 180);
        const stickmanHeight = 4;

        const modelBox = new THREE.Box3().setFromObject(stickmanModel);
        const topOfHeadY = modelBox.max.y * stickmanHeight;
        const frontOffsetZ = modelBox.max.z * stickmanHeight;
        const nameplateOffset = 1.5;

        const startAngle = -sectorAngleRadians / 2;
        const angleIncrement = numStickmen > 1 ? sectorAngleRadians / (numStickmen - 1) : 0;

        for (let i = 0; i < numStickmen; i++) {
          const stickman = SkeletonUtils.clone(stickmanModel);
          const angle = startAngle + i * angleIncrement;

          const x = radius * Math.sin(angle);
          const z = radius * Math.cos(angle);
          const y = stickmanHeight / 2;
          stickman.position.set(x, y, z);

          stickman.scale.multiplyScalar(stickmanHeight);
          stickman.lookAt(new THREE.Vector3(0, y, 0));

          const nameplate = this._createNameplate(names[i], URLS[brands[i]]);
          nameplate.position.set(0, topOfHeadY + nameplateOffset, -frontOffsetZ);
          stickman.add(nameplate);

          stickmanGroup.add(stickman);

          if (idleClip) {
            const mixer = new THREE.AnimationMixer(stickman);
            const action = mixer.clipAction(idleClip);
            action.time = Math.random() * idleClip.duration;
            action.play();
            mixer.update(0);
          }
        }
        
        this._scene.add(stickmanGroup);
        this._FrameGroup(stickmanGroup);
      },
      undefined,
      (error) => {
        console.error("Error loading FBX model:", error);
      }
    );
  }

  _createNameplate(name, imageUrl) {
    const container = document.createElement('div');
    container.style.backgroundColor = 'rgba(255, 255, 255, 0)';
    container.style.padding = '8px 12px';
    container.style.borderRadius = '8px';
    container.style.display = 'flex';
    container.style.alignItems = 'center';
    container.style.justifyContent = 'center';
    container.style.gap = '10px';
    container.style.textAlign = 'center';

    const img = document.createElement('img');
    img.src = imageUrl;
    img.style.width = '60px';
    img.style.height = '60px';
    img.style.borderRadius = '80%';
    img.style.objectFit = 'cover';
    img.style.backgroundColor = 'white';

    const text = document.createElement('div');
    text.textContent = name;
    text.style.color = 'white';
    text.style.fontFamily = 'Arial, sans-serif';
    text.style.fontSize = '16px';

    container.appendChild(img);
    container.appendChild(text);

    const label = new CSS2DObject(container);
    return label;
  }

  _FrameGroup(group) {
    const box = new THREE.Box3().setFromObject(group);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());

    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = this._camera.fov * (Math.PI / 180);
    
    let cameraZ = Math.abs(maxDim / Math.tan(fov / 2));
    cameraZ *= 0.5; 

    this._camera.position.set(center.x, center.y, center.z - cameraZ);

    const shiftY = size.y / 2.5; 
    this._camera.position.y += shiftY;

    const newTarget = center.clone();
    newTarget.y += shiftY;
    this._controls.target.copy(newTarget);
    this._controls.update();
  }

  _NormalizeModel(model) {
    const box = new THREE.Box3().setFromObject(model);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    model.position.copy(center).negate();
    const wrapper = new THREE.Group();
    wrapper.add(model);
    const scale = 1.0 / size.y;
    wrapper.scale.set(scale, scale, scale);
    return wrapper;
  }

  _OnWindowResize() {
    this._camera.aspect = window.innerWidth / window.innerHeight;
    this._camera.updateProjectionMatrix();
    this._threejs.setSize(window.innerWidth, window.innerHeight);
    this._labelRenderer.setSize(window.innerWidth, window.innerHeight);
  }

  _RAF() {
    requestAnimationFrame(() => {
      this._threejs.render(this._scene, this._camera);
      this._labelRenderer.render(this._scene, this._camera);
      this._RAF();
    });
  }
}


let _APP = null;

window.addEventListener('DOMContentLoaded', () => {
  _APP = new BasicWorldDemo();
});