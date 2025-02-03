## Important things you need to know before you start

Locations and geometry of electrodes of Neuropixels 1.0 and 2.0 probes are pre-defined in HERBS. 


<p style="font-size:120%; color:red">When ONLY atlas is loaded, the pre-surgery mode is ON.</p>
<p style="font-size:120%; color:red">When BOTH atlas and histological images are loaded, the after-surgery mode is ON.</p>

Therefore, if users would like to plan the coordinates before surgeries, the histological images should not be loaded.

To work with probe related projects, 
the most important button one needs to know is <b style="font-size:120%; color:blue">Probe Marker</b> button.
<p align="center">
<img src="../../image/probe_related/probe_marker.jpg" width="100%">
</p>

After this button is checked, a group of sub-tools pop up: 
<b>Multi-shanks Switch</b> button, <b>Probe Color</b> selector, <b>Probe Type</b> combobox,
<b>Merge sites</b> button and <b>Sites Face Direction</b> combobox.
<p align="center">
<img src="../../image/probe_related/im1.jpg" width="100%">
</p>
<p align="center">
<img src="../../image/probe_related/im2.jpg" width="100%">
</p>
<p align="center">
<img src="../../image/probe_related/im3.jpg" width="100%">
</p>
<p align="center">
<img src="../../image/probe_related/im4.jpg" width="100%">
</p>
<p align="center">
<img src="../../image/probe_related/im5.jpg" width="100%">
</p>

<p style="font-size:120%; color:red">The Probe Type defines the geometry of a single shank of a probe.</p>

For example, the current version of Neuropixels 2.0 is a 4-shnaks probe. Choosing <em>Neuropixel 2.0</em> 
in the <b>Probe Type</b> combobox will turn on the single shank Neuropixel 2.0.

To turn on the 4-shanks version, users need to switch <b>Multi-shanks Switch</b> button checked. 

The probe is considered to be a 3D rigid object and electrodes are on one surface of the probe. 
When we clicked the trajectory of a probe, we assume the trajectory is the probe center.
<p align="center">
<img src="../../image/probe_related/probe.jpg" width="50%">
</p>

The <b>Sites Face Direction</b> combobox is different for pre-planning (left) and after-reconstruction (right).
<p align="center">
<img src="../../image/probe_related/pre_face.jpg" width="30%">
<img src="../../image/probe_related/after_face.jpg" width="30%">
</p>

For pre-planning, the 'Out' indicates the electrodes facing to the users when looking at a 2D section window, 
whereas 'in' indicates the electrodes facing away from the users. 
'Left' and 'Right' indicates facing to the left-hand side and right-hand side of the section window.
<p align="center">
<img src="../../image/probe_related/out_face.jpg" width="40%">
<img src="../../image/probe_related/in_face.jpg" width="40%">
</p>

For after-reconstruction, we consider only the relation between the user and the probe when he/she is doing the surgery.
So far we give 4 options depending on how users holding the probe when they are doing the surgery.
<p align="center">
<img src="../../image/probe_related/after_face_man.jpg" width="100%">
</p>