export class ParticleSystem {
  constructor(scene, THREE) {
    this.scene = scene;
    this.THREE = THREE;
    
    this.particles = null;
    this.particleMaterial = null;

    this.init();
  }

  init() {
    const particleCount = 300;
    const particles = new this.THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    const sizes = new Float32Array(particleCount);

    for (let i = 0; i < particleCount; i++) {
      const i3 = i * 3;
      positions[i3] = (Math.random() - 0.5) * 80;
      positions[i3 + 1] = Math.random() * 30 + 5;
      positions[i3 + 2] = (Math.random() - 0.5) * 80;

      const hue = Math.random() * 0.2 + 0.55;
      const color = new this.THREE.Color().setHSL(hue, 0.3, 0.4);
      colors[i3] = color.r;
      colors[i3 + 1] = color.g;
      colors[i3 + 2] = color.b;

      sizes[i] = Math.random() * 1.5 + 0.3;
    }

    particles.setAttribute('position', new this.THREE.BufferAttribute(positions, 3));
    particles.setAttribute('color', new this.THREE.BufferAttribute(colors, 3));
    particles.setAttribute('size', new this.THREE.BufferAttribute(sizes, 1));

    this.particleMaterial = new this.THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0 },
      },
      vertexShader: `
            attribute float size;
            attribute vec3 color;
            varying vec3 vColor;
            uniform float time;
            
            void main() {
              vColor = color;
              vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
              gl_PointSize = size * (300.0 / -mvPosition.z) * (1.0 + sin(time * 2.0 + position.x * 0.1) * 0.3);
              gl_Position = projectionMatrix * mvPosition;
            }
          `,
      fragmentShader: `
            varying vec3 vColor;
            
            void main() {
              float dist = distance(gl_PointCoord, vec2(0.5));
              if (dist > 0.5) discard;
              
              float alpha = 1.0 - (dist * 2.0);
              alpha *= alpha; // Softer edges
              
              gl_FragColor = vec4(vColor, alpha * 0.3); 
            }
          `,
      transparent: true,
      blending: this.THREE.AdditiveBlending,
      depthWrite: false,
    });

    this.particles = new this.THREE.Points(particles, this.particleMaterial);
    this.scene.add(this.particles);
  }

  update(time, phaseValue) {
    if (this.particleMaterial) {
      this.particleMaterial.uniforms.time.value = time * 0.001;
    }

    if (this.particles) {
      this.particles.rotation.y = time * 0.0001 * (1 - phaseValue * 0.5);

      const positions = this.particles.geometry.attributes.position.array;
      for (let i = 0; i < positions.length; i += 3) {
        const movementScale = 1 - phaseValue * 0.5;
        positions[i + 1] += Math.sin(time * 0.001 + positions[i] * 0.01) * 0.02 * movementScale;
        if (positions[i + 1] < 0) {
          positions[i + 1] = 35;
        }
      }
      this.particles.geometry.attributes.position.needsUpdate = true;

      if (this.particles.geometry.attributes.color) {
        const colors = this.particles.geometry.attributes.color.array;
        for (let i = 0; i < colors.length; i += 3) {
          const baseHue = 0.5 + Math.random() * 0.3;
          const phaseShift = phaseValue * 0.1;
          const hue = baseHue + phaseShift;
          const saturation = 0.8 - phaseValue * 0.2;
          const lightness = 0.6 - phaseValue * 0.2;

          const color = new this.THREE.Color().setHSL(hue, saturation, lightness);
          colors[i] = color.r;
          colors[i + 1] = color.g;
          colors[i + 2] = color.b;
        }
        this.particles.geometry.attributes.color.needsUpdate = true;
      }
    }
  }
}
