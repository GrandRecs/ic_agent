import { useGLTF } from '@react-three/drei';
import React, { useRef } from "react";
import { GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { Group } from 'three';

type GLTFObject = GLTF & {
    nodes: {
        [name: string]: THREE.Mesh;
    };
    materials: {
        [name: string]: THREE.Material;
    };
};

export default function Earth(props: { setMesh: (mesh: any) => void }) {
    const group = useRef<Group>(null);
    const { nodes, materials } = useGLTF("Earth_1_12756.glb") as unknown as GLTFObject;
    return (
        <group ref={group} dispose={null} scale={0.013}>
            <mesh ref={mesh => {
                props.setMesh(mesh);
            }} geometry={nodes.Cube001.geometry} material={materials['Default OBJ']}/>
        </group>
    )
}

useGLTF.preload("Earth_1_12756.glb");