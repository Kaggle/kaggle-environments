export class LightingManager {
  constructor(scene, THREE, renderer, assetManager) {
    this.scene = scene;
    this.THREE = THREE;
    this.assetManager = assetManager;
    this.pmremGenerator = new THREE.PMREMGenerator(renderer);
    this.pmremGenerator.compileEquirectangularShader();

    this.ambientLight = null;
    this.rimLight = null;
    this.hemiLight = null;
    this.fillLight = null;

    this.dayEnvMap = null;
    this.nightEnvMap = null;

    this.init();
  }

  init() {
    // Ambient Light
    this.ambientLight = new this.THREE.AmbientLight(0x4a4a3a, 0.5);
    this.ambientLight.name = 'ambientLight';
    this.scene.add(this.ambientLight);

    // Rim Light
    this.rimLight = new this.THREE.DirectionalLight(0xaa6633, 0);
    this.rimLight.position.set(-20, 10, -30);
    this.scene.add(this.rimLight);

    // Hemisphere Light
    this.hemiLight = new this.THREE.HemisphereLight(0x6a7a9a, 0x3a2a1a, 0.5);
    this.scene.add(this.hemiLight);

    // Fill Light
    this.fillLight = new this.THREE.DirectionalLight(0x5a4a3a, 0.15);
    this.fillLight.position.set(0, -1, 0);
    this.scene.add(this.fillLight);

    this.loadHDRIs();
  }

  loadHDRIs() {
    const dayPath = `${import.meta.env.BASE_URL}static/werewolf/hdri/greenwich_park_1k.hdr`;
    const nightPath = `${import.meta.env.BASE_URL}static/werewolf/hdri/rogland_clear_night_1k.hdr`;

    this.assetManager.loadHDR(dayPath).then((texture) => {
        const envMap = this.pmremGenerator.fromEquirectangular(texture).texture;
        this.dayEnvMap = envMap;
        
        // Set initial if day
        if (!this.scene.environment) this.scene.environment = envMap;
    });

    this.assetManager.loadHDR(nightPath).then((texture) => {
        const envMap = this.pmremGenerator.fromEquirectangular(texture).texture;
        this.nightEnvMap = envMap;
    });
  }

  update(phase) {
    // Switch Environment Map based on phase
    if (phase > 0.5 && this.nightEnvMap) {
        if (this.scene.environment !== this.nightEnvMap) {
            this.scene.environment = this.nightEnvMap;
        }
    } else if (phase <= 0.5 && this.dayEnvMap) {
        if (this.scene.environment !== this.dayEnvMap) {
            this.scene.environment = this.dayEnvMap;
        }
    }

    if (this.rimLight) {
      const nightColor = new this.THREE.Color(0x664422);
      const dayColor = new this.THREE.Color(0xaa6633);
      this.rimLight.color.copy(dayColor).lerp(nightColor, phase);
      this.rimLight.intensity = 0.3 - phase * 0.1;
    }

    if (this.hemiLight) {
      const nightSkyColor = new this.THREE.Color(0x2a2a4a);
      const daySkyColor = new this.THREE.Color(0x6a7a9a);
      const nightGroundColor = new this.THREE.Color(0x2a1a0a);
      const dayGroundColor = new this.THREE.Color(0x3a2a1a);

      this.hemiLight.color.copy(daySkyColor).lerp(nightSkyColor, phase);
      this.hemiLight.groundColor.copy(dayGroundColor).lerp(nightGroundColor, phase);
      this.hemiLight.intensity = 0.3 - phase * 0.1;
    }

    if (this.ambientLight) {
      const nightColor = new this.THREE.Color(0x3a3a5a);
      const dayColor = new this.THREE.Color(0x9a9a8a);
      this.ambientLight.color.copy(dayColor).lerp(nightColor, phase);
      this.ambientLight.intensity = 0.4 - phase * 0.2;
    }
  }
}
