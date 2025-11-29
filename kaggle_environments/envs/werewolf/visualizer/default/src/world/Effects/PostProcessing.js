export class PostProcessing {
  constructor(scene, camera, renderer, width, height, THREE, EffectComposer, RenderPass, UnrealBloomPass, ShaderPass, FilmPass) {
    this.scene = scene;
    this.camera = camera;
    this.renderer = renderer;
    this.width = width;
    this.height = height;
    this.THREE = THREE;
    
    // Modules
    this.EffectComposer = EffectComposer;
    this.RenderPass = RenderPass;
    this.UnrealBloomPass = UnrealBloomPass;
    this.ShaderPass = ShaderPass;
    this.FilmPass = FilmPass;

    this.composer = null;
    this.bloomPass = null;
    this.filmPass = null;
    this.atmospherePass = null;

    this.init();
  }

  init() {
    this.composer = new this.EffectComposer(this.renderer);

    const renderPass = new this.RenderPass(this.scene, this.camera);
    this.composer.addPass(renderPass);

    this.bloomPass = new this.UnrealBloomPass(
      new this.THREE.Vector2(this.width, this.height),
      0.15,
      0.4,
      0.85
    );
    this.composer.addPass(this.bloomPass);

    const atmosphereShader = {
      uniforms: {
        'tDiffuse': { value: null },
        'time': { value: 0.0 },
        'phase': { value: 0.0 },
      },
      vertexShader: `
            varying vec2 vUv;
            void main() {
              vUv = uv;
              gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
          `,
      fragmentShader: `
            uniform sampler2D tDiffuse;
            uniform float time;
            uniform float phase;
            varying vec2 vUv;
            
            void main() {
              vec4 color = texture2D(tDiffuse, vUv);
              if (phase > 0.5) {
                // Night Phase - Blood Moon Theme
                // Shift towards red/purple, increase contrast
                vec3 nightTint = vec3(1.1, 0.8, 0.9); // Reddish tint
                color.rgb = mix(color.rgb, color.rgb * nightTint, 0.4);
                
                // Enhance reds specifically
                float redBoost = color.r * 0.2;
                color.r += redBoost;
                
                // Darken slightly for mood
                color.rgb *= 0.9;
                
                color.rgb = pow(color.rgb, vec3(1.1)); // Contrast
              } else {
                // Day Phase
                color.rgb = mix(color.rgb, color.rgb * vec3(1.05, 1.0, 0.95), 0.2);
              }
              
              vec2 center = vec2(0.5, 0.5);
              float dist = distance(vUv, center);
              float vignette = 1.0 - smoothstep(0.4, 1.0, dist);
              color.rgb *= mix(0.7, 1.0, vignette);
              gl_FragColor = color;
            }
          `,
    };

    this.atmospherePass = new this.ShaderPass(atmosphereShader);
    this.composer.addPass(this.atmospherePass);
  }

  update(time, phaseValue) {
    if (this.bloomPass) {
      this.bloomPass.strength = 0.1 + phaseValue * 0.03;
      this.bloomPass.radius = 0.08 + phaseValue * 0.04;
      this.bloomPass.threshold = 0.0 + phaseValue * 0.1;
    }

    if (this.atmospherePass) {
        this.atmospherePass.uniforms.time.value = time * 0.001;
        this.atmospherePass.uniforms.phase.value = phaseValue;
    }
  }

  render() {
    this.composer.render();
  }
}
