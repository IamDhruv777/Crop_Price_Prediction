"""
Relatable 3D Mandi Tarazu (Weighing Scale) & Golden Wheat Harvest WebGL Animation.
Directly relatable to Indian farmers, village vendors, and mandi traders as the
universal symbol of fair price discovery and honest weight.
"""

def render_3d_tarazu(is_hi: bool = False, app_title: str = "Krishi Bhav Saathi") -> str:
    sub_text_hi = "खेत से दुकान और मंडी तक — सही तौल और सही दाम"
    sub_text_en = "From Farm to Shop & Mandi — Fair Weight & True Price"
    sub_text = sub_text_hi if is_hi else sub_text_en

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Mandi Tarazu 3D</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@600;800;900&family=Poppins:wght@400;600&display=swap');

            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                user-select: none;
            }}

            body, html {{
                width: 100%;
                height: 100%;
                overflow: hidden;
                background: radial-gradient(circle at 50% 30%, #133a26 0%, #081e14 60%, #030d08 100%);
                font-family: 'Outfit', 'Poppins', sans-serif;
            }}

            #canvas-container {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: 1;
            }}

            .hero-caption {{
                position: absolute;
                bottom: 12px;
                left: 0;
                width: 100%;
                text-align: center;
                z-index: 2;
                pointer-events: none;
            }}

            .caption-pill {{
                display: inline-block;
                background: rgba(4, 16, 12, 0.7);
                backdrop-filter: blur(8px);
                border: 1px solid rgba(245, 158, 11, 0.4);
                color: #fde68a;
                font-size: 0.95rem;
                font-weight: 600;
                padding: 6px 20px;
                border-radius: 30px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.4);
                letter-spacing: 0.3px;
            }}
        </style>
    </head>
    <body>
        <div id="canvas-container"></div>
        <div class="hero-caption">
            <span class="caption-pill">⚖️ {sub_text}</span>
        </div>

        <script>
            // --- Three.js Relatable 3D Mandi Weighing Scale (Tarazu) ---
            const container = document.getElementById('canvas-container');
            const scene = new THREE.Scene();

            const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 1, 1000);
            camera.position.set(0, 4, 38);

            const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            renderer.shadowMap.enabled = true;
            container.appendChild(renderer.domElement);

            // Lighting: Warm golden sunlight & soft green ground bounce
            const ambient = new THREE.AmbientLight(0xfff7ed, 0.9);
            scene.add(ambient);

            const sunLight = new THREE.DirectionalLight(0xf59e0b, 2.2);
            sunLight.position.set(15, 30, 20);
            scene.add(sunLight);

            const fillLight = new THREE.PointLight(0x34d399, 1.5, 60);
            fillLight.position.set(-20, -10, 15);
            scene.add(fillLight);

            // --- 3D Traditional Brass Mandi Scale (Tarazu) Group ---
            const scaleGroup = new THREE.Group();
            scene.add(scaleGroup);

            const brassMaterial = new THREE.MeshStandardMaterial({{
                color: 0xd97706,
                metalness: 0.85,
                roughness: 0.25
            }});

            const goldGlowMaterial = new THREE.MeshStandardMaterial({{
                color: 0xfbbf24,
                metalness: 0.9,
                roughness: 0.2,
                emissive: 0xd97706,
                emissiveIntensity: 0.25
            }});

            const sackMaterial = new THREE.MeshStandardMaterial({{
                color: 0xb45309,
                roughness: 0.85
            }});

            // 1. Central Brass Stand
            const baseGeo = new THREE.CylinderGeometry(5.5, 6.5, 1.2, 32);
            const baseMesh = new THREE.Mesh(baseGeo, brassMaterial);
            baseMesh.position.y = -10;
            scaleGroup.add(baseMesh);

            const pillarGeo = new THREE.CylinderGeometry(0.65, 0.9, 18, 24);
            const pillarMesh = new THREE.Mesh(pillarGeo, brassMaterial);
            pillarMesh.position.y = -1;
            scaleGroup.add(pillarMesh);

            const topKnobGeo = new THREE.SphereGeometry(1.2, 24, 24);
            const topKnob = new THREE.Mesh(topKnobGeo, goldGlowMaterial);
            topKnob.position.y = 8;
            scaleGroup.add(topKnob);

            // Fulcrum Ring
            const ringGeo = new THREE.TorusGeometry(1.4, 0.25, 16, 32);
            const ringMesh = new THREE.Mesh(ringGeo, brassMaterial);
            ringMesh.position.y = 9.5;
            scaleGroup.add(ringMesh);

            // 2. Balancing Cross Beam (Rotates dynamically)
            const beamGroup = new THREE.Group();
            beamGroup.position.y = 7.8;
            scaleGroup.add(beamGroup);

            const beamGeo = new THREE.CylinderGeometry(0.35, 0.35, 24, 16);
            const beamMesh = new THREE.Mesh(beamGeo, brassMaterial);
            beamMesh.rotation.z = Math.PI / 2;
            beamGroup.add(beamMesh);

            // Beam End Rings
            const leftEndKnob = new THREE.Mesh(new THREE.SphereGeometry(0.6, 16, 16), goldGlowMaterial);
            leftEndKnob.position.x = -12;
            beamGroup.add(leftEndKnob);

            const rightEndKnob = new THREE.Mesh(new THREE.SphereGeometry(0.6, 16, 16), goldGlowMaterial);
            rightEndKnob.position.x = 12;
            beamGroup.add(rightEndKnob);

            // 3. Left Balance Pan (Holds Golden Crop Sack)
            const leftPanGroup = new THREE.Group();
            leftPanGroup.position.set(-12, -7.5, 0);
            beamGroup.add(leftPanGroup);

            // Strings
            const stringMat = new THREE.LineBasicMaterial({{ color: 0xfde68a, transparent: true, opacity: 0.8 }});
            function createStrings(targetGroup) {{
                for (let a = 0; a < 3; a++) {{
                    const angle = (a * (Math.PI * 2)) / 3;
                    const sx = Math.cos(angle) * 3.8;
                    const sz = Math.sin(angle) * 3.8;
                    const pts = [new THREE.Vector3(0, 7.5, 0), new THREE.Vector3(sx, 0, sz)];
                    const strGeo = new THREE.BufferGeometry().setFromPoints(pts);
                    targetGroup.add(new THREE.Line(strGeo, stringMat));
                }}
            }}
            createStrings(leftPanGroup);

            // Pan Dish
            const panGeo = new THREE.CylinderGeometry(4.2, 2.5, 0.8, 32);
            const leftPanDish = new THREE.Mesh(panGeo, brassMaterial);
            leftPanGroup.add(leftPanDish);

            // Golden Grain Sacks in Left Pan
            const sackGeo = new THREE.SphereGeometry(2.2, 16, 16);
            sackGeo.scale(1, 1.4, 1);
            const sack1 = new THREE.Mesh(sackGeo, sackMaterial);
            sack1.position.set(0, 2.2, 0);
            leftPanGroup.add(sack1);

            const wheatPiles = new THREE.Mesh(new THREE.ConeGeometry(2.8, 2, 16), goldGlowMaterial);
            wheatPiles.position.set(0, 1.5, 0);
            leftPanGroup.add(wheatPiles);

            // 4. Right Balance Pan (Holds Golden Coins / Profit)
            const rightPanGroup = new THREE.Group();
            rightPanGroup.position.set(12, -7.5, 0);
            beamGroup.add(rightPanGroup);
            createStrings(rightPanGroup);

            const rightPanDish = new THREE.Mesh(panGeo, brassMaterial);
            rightPanGroup.add(rightPanDish);

            // Golden Coins / Weights in Right Pan
            for (let c = 0; c < 5; c++) {{
                const coin = new THREE.Mesh(new THREE.CylinderGeometry(1.6 - c * 0.2, 1.6 - c * 0.2, 0.5, 24), goldGlowMaterial);
                coin.position.set(0, 0.5 + c * 0.5, 0);
                rightPanGroup.add(coin);
            }}

            // 5. Floating Harvest Sparkles / Golden Chaff (500 particles)
            const pCount = 500;
            const pGeo = new THREE.BufferGeometry();
            const pPos = new Float32Array(pCount * 3);
            for (let i = 0; i < pCount; i++) {{
                pPos[i * 3] = (Math.random() - 0.5) * 60;
                pPos[i * 3 + 1] = (Math.random() - 0.5) * 40;
                pPos[i * 3 + 2] = (Math.random() - 0.5) * 30;
            }}
            pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
            const pMat = new THREE.PointsMaterial({{
                color: 0xfbbf24,
                size: 0.6,
                transparent: true,
                opacity: 0.75,
                blending: THREE.AdditiveBlending
            }});
            const pSystem = new THREE.Points(pGeo, pMat);
            scene.add(pSystem);

            // 6. Interactive Mouse Movement & Parallax
            let targetRotZ = 0;
            let currentRotZ = 0;
            let mouseX = 0;
            let mouseY = 0;

            window.addEventListener('mousemove', (e) => {{
                const nx = (e.clientX / window.innerWidth) * 2 - 1;
                const ny = (e.clientY / window.innerHeight) * 2 - 1;
                mouseX = nx;
                mouseY = ny;
                // Move balance beam based on mouse horizontal movement
                targetRotZ = -nx * 0.16;
            }});

            // 7. Animation Loop
            let clock = new THREE.Clock();
            function animate() {{
                requestAnimationFrame(animate);
                const t = clock.getElapsedTime();

                // Gentle organic breathing balance
                const naturalSway = Math.sin(t * 1.5) * 0.05;
                currentRotZ += (targetRotZ + naturalSway - currentRotZ) * 0.04;
                beamGroup.rotation.z = currentRotZ;

                // Counter-rotate balance pans so they hang vertically down
                leftPanGroup.rotation.z = -currentRotZ;
                rightPanGroup.rotation.z = -currentRotZ;

                // Subtle scale turn on Y axis
                scaleGroup.rotation.y = Math.sin(t * 0.6) * 0.18 + mouseX * 0.35;
                scaleGroup.rotation.x = mouseY * 0.12;

                // Particle rise
                const positions = pGeo.attributes.position.array;
                for (let i = 1; i < pCount * 3; i += 3) {{
                    positions[i] += 0.04;
                    if (positions[i] > 20) positions[i] = -20;
                }}
                pGeo.attributes.position.needsUpdate = true;

                renderer.render(scene, camera);
            }}
            animate();

            // Resize handling
            window.addEventListener('resize', () => {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }});
        </script>
    </body>
    </html>
    """
