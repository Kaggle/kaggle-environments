export class SceneManager {
  constructor(options, THREE, OrbitControls, CSS2DRenderer) {
    this.THREE = THREE;
    this.width = options.width;
    this.height = options.height;
    this.parent = options.parent;

    this.renderer = null;
    this.labelRenderer = null;
    this.camera = null;
    this.scene = null;
    this.controls = null;

    this.init(OrbitControls, CSS2DRenderer);
  }

  init(OrbitControls, CSS2DRenderer) {
    // WebGL Renderer
    this.renderer = new this.THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
    });
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = this.THREE.PCFSoftShadowMap;
    this.renderer.shadowMap.autoUpdate = true;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(this.width, this.height);
    this.renderer.outputEncoding = this.THREE.sRGBEncoding;
    this.renderer.toneMapping = this.THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;

    this.renderer.domElement.style.position = 'absolute';
    this.renderer.domElement.style.top = '0';
    this.renderer.domElement.style.left = '0';
    this.renderer.domElement.style.zIndex = '0';
    this.parent.appendChild(this.renderer.domElement);

    // CSS2D Renderer
    this.labelRenderer = new CSS2DRenderer();
    this.labelRenderer.setSize(this.width, this.height);
    this.labelRenderer.domElement.style.position = 'absolute';
    this.labelRenderer.domElement.style.top = '0px';
    this.labelRenderer.domElement.style.left = '0px';
    this.labelRenderer.domElement.style.zIndex = '1';
    this.labelRenderer.domElement.style.pointerEvents = 'none';
    this.parent.appendChild(this.labelRenderer.domElement);

    // Camera
    const fov = 60;
    const aspect = this.width / this.height;
    const near = 1.0;
    const far = 100000.0;
    this.camera = new this.THREE.PerspectiveCamera(fov, aspect, near, far);
    this.camera.position.set(0, 0, 50);

    // Scene
    this.scene = new this.THREE.Scene();

    // Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0, 0);
    this.controls.enableKeys = false;
    this.controls.update();
  }

  update() {
    this.controls.update();
  }
}
