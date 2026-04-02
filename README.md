# Vector Field K-Means

This is a translation of the original VFKM algorithm from C++ to Python wich also includes new features as a Visualizer.

## (Adapted) abstract

> To understand trends in movement patterns, analyzing this data and discover more underlying patterns,
we introduce a novel technique which we call ***Vector-Field K-Means***.


> The central idea of our approach is to use vector fields to induce a similarity notion between trajectories.
Our approach is based on the premise that movement trends in trajectory data can be modeled as flows within multiple vector fields,
and the vector field itself is what defines each of the clusters.
We also show how VFKM connects techniques for scalar field design on meshes and k-means clustering.
We present an algorithm that finds a locally optimal clustering of trajectories into vector fields,
and demonstrate how vector-field k-means can be used to mine patterns from trajectory data.


- (Adapted) abstract from [Vector Field k-Means: Clustering Trajectories by Fitting Multiple Vector Fields](https://arxiv.org/pdf/1208.5801)


## DEMO

### DataSet
<img width="1920" height="1440" alt="dataset" src="https://github.com/user-attachments/assets/42764a22-a028-45b8-b355-3575afe8a333" />

### Clusters found

<div style="display: flex; gap: 10px;">
  <img width="48%" alt="cluster_1" src="https://github.com/user-attachments/assets/a36146f5-b24f-4d0f-9d0e-eb99c4912dbf" />
  <img width="48%" alt="cluster_0" src="https://github.com/user-attachments/assets/2915d09a-c8c2-48d3-a4a4-5653e91fa04b" />
</div>

