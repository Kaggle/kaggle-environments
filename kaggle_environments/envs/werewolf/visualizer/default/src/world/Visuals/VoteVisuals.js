export class VoteVisuals {
  constructor(scene, THREE, votingArcsGroup, targetRingsGroup) {
    this.scene = scene;
    this.THREE = THREE;
    this.votingArcsGroup = votingArcsGroup;
    this.targetRingsGroup = targetRingsGroup;
    
    this.activeVoteArcs = new Map();
    this.activeTargetRings = new Map();
    this.animatingTrails = [];
    this.particlePool = [];

    // Shared Resources
    this.ringGeometry = new this.THREE.RingGeometry(2, 2.2, 32);
    this.baseRingMaterial = new this.THREE.MeshBasicMaterial({
      color: 0x00ffff,
      transparent: true,
      opacity: 0,
      side: this.THREE.DoubleSide,
    });
    this.particleMaterials = new Map();
  }

  getParticleMaterial(color) {
    if (!this.particleMaterials.has(color)) {
      const mat = new this.THREE.PointsMaterial({
        color: color,
        size: 0.3,
        transparent: true,
        opacity: 0.8,
        blending: this.THREE.AdditiveBlending,
        sizeAttenuation: true,
      });
      this.particleMaterials.set(color, mat);
    }
    return this.particleMaterials.get(color);
  }

  getPooledParticles(color) {
      let particles;
      if (this.particlePool.length > 0) {
          particles = this.particlePool.pop();
      } else {
          const particleCount = 50;
          const geometry = new this.THREE.BufferGeometry();
          const positions = new Float32Array(particleCount * 3);
          geometry.setAttribute('position', new this.THREE.BufferAttribute(positions, 3));
          particles = new this.THREE.Points(geometry, this.getParticleMaterial(color));
      }
      particles.material = this.getParticleMaterial(color);
      return particles;
  }

  returnToPool(particles) {
      if (particles.parent) {
          particles.parent.remove(particles);
      }
      this.particlePool.push(particles);
  }

  createVoteParticleTrail(voter, target, voterName, targetName, color = 0x00ffff) {
    if (!voter || !target) return;

    const startPos = voter.container.position.clone();
    startPos.y += 1.5;
    const endPos = target.container.position.clone();
    endPos.y += 1.5;

    const midPos = new this.THREE.Vector3().addVectors(startPos, endPos).multiplyScalar(0.5);
    const dist = startPos.distanceTo(endPos);
    midPos.y += dist * 0.3;

    const curve = new this.THREE.CatmullRomCurve3([startPos, midPos, endPos]);
    const particleCount = 50;
    
    const particles = this.getPooledParticles(color);
    this.votingArcsGroup.add(particles);

    const trail = {
      particles,
      curve,
      target: targetName,
      startTime: Date.now(),
      update: () => {
        const elapsedTime = (Date.now() - trail.startTime) / 1000;
        const positions = trail.particles.geometry.attributes.position.array;
        for (let i = 0; i < particleCount; i++) {
          const t = (elapsedTime * 0.2 + i / particleCount) % 1;
          const pos = trail.curve.getPointAt(t);
          positions[i * 3] = pos.x;
          positions[i * 3 + 1] = pos.y;
          positions[i * 3 + 2] = pos.z;
        }
        trail.particles.geometry.attributes.position.needsUpdate = true;
      },
    };
    this.activeVoteArcs.set(voterName, trail);
    this.animatingTrails.push(trail);
  }

  updateTargetRing(target, targetName, voteCount) {
    if (!target) return;

    let ringData = this.activeTargetRings.get(targetName);

    if (voteCount > 0 && !ringData) {
      const ring = new this.THREE.Mesh(this.ringGeometry, this.baseRingMaterial.clone());
      ring.position.copy(target.container.position);
      ring.position.y = 0.1;
      ring.rotation.x = -Math.PI / 2;

      this.targetRingsGroup.add(ring);
      ringData = { ring, material: ring.material, targetOpacity: 0 };
      this.activeTargetRings.set(targetName, ringData);
    }

    if (ringData) {
      if (voteCount > 0) {
        ringData.targetOpacity = 0.3 + Math.min(voteCount * 0.2, 0.7);
      } else {
        ringData.targetOpacity = 0;
      }
    }
  }

  updateVoteVisuals(votes, playerObjects, clearAll = false) {
    if (!playerObjects || playerObjects.size === 0) return;

    if (clearAll) {
      votes.clear();
    }

    this.activeVoteArcs.forEach((trail, voterName) => {
      if (!votes.has(voterName)) {
        this.returnToPool(trail.particles);
        this.activeVoteArcs.delete(voterName);
        this.animatingTrails = this.animatingTrails.filter((t) => t !== trail);
      }
    });

    votes.forEach((voteData, voterName) => {
      const { target: targetName, type } = voteData;
      const existingTrail = this.activeVoteArcs.get(voterName);

      let color = 0x00ffff;
      if (type === 'night_vote') color = 0xff0000;
      else if (type === 'doctor_heal_action') color = 0x00ff00;
      else if (type === 'seer_inspection') color = 0x800080;

      if (existingTrail) {
        if (existingTrail.target !== targetName) {
          this.returnToPool(existingTrail.particles);
          this.animatingTrails = this.animatingTrails.filter((t) => t !== existingTrail);
          this.createVoteParticleTrail(playerObjects.get(voterName), playerObjects.get(targetName), voterName, targetName, color);
        }
      } else {
        this.createVoteParticleTrail(playerObjects.get(voterName), playerObjects.get(targetName), voterName, targetName, color);
      }
    });

    const targetVoteCounts = new Map();
    votes.forEach((voteData) => {
      const { target: targetName } = voteData;
      targetVoteCounts.set(targetName, (targetVoteCounts.get(targetName) || 0) + 1);
    });

    playerObjects.forEach((player, playerName) => {
      this.updateTargetRing(player, playerName, targetVoteCounts.get(playerName) || 0);
    });
  }

  update() {
    this.animatingTrails.forEach((trail) => trail.update());

    this.activeTargetRings.forEach((ringData, targetName) => {
      const diff = ringData.targetOpacity - ringData.material.opacity;
      if (Math.abs(diff) > 0.01) {
        ringData.material.opacity += diff * 0.1;
      } else if (ringData.targetOpacity === 0 && ringData.material.opacity > 0) {
        this.targetRingsGroup.remove(ringData.ring);
        ringData.material.dispose(); // Dispose material clone
        this.activeTargetRings.delete(targetName);
      }
    });
  }
}
