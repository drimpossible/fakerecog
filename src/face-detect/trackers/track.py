import torch
from trackers.sort import Sort, iou
import numpy as np

# Maybe break this function down into smaller functions? Only useful if we compare multiple trackers.
def get_tracks(opt, bboxes, landmarks, confidence, frames):
    mot_tracker = Sort(max_age=opt.max_track_age, min_hits=opt.min_track_hits)
    num_frames = frames.max()
    trackids, smooth_bboxes = torch.zeros_like(frames)-1, torch.zeros_like(bboxes)-1
    passed_det = []
    for frame in range(num_frames):
        idx = torch.nonzero(frames==frame,as_tuple=True)[0]
        det = bboxes[idx]
        conf = confidence[idx]
        det_with_conf = torch.cat((det, conf.unsqueeze(1)), dim=1)
        smdet_with_tracks = torch.from_numpy(mot_tracker.update(det_with_conf.numpy()))
        
        for i in range(smdet_with_tracks.shape[0]):
            # Match the detected tracklet with original tracklet. Don't know why SORT outputs this way :/
            maxarea, mapid = -1, -1
            for j in range(det.shape[0]):
                area = iou(det[j],smdet_with_tracks[i])
                if area>maxarea:
                    mapid = idx[j]
                    maxarea = area
            passed_det.append(mapid.item())
            trackids[mapid] = smdet_with_tracks[i,4].long()
            smooth_bboxes[mapid] = smdet_with_tracks[i,:4].float()
    
    uniq_tracks, counts = torch.unique(trackids, return_counts=True)

    # Warning: Hacky work ahead
    dumpster = (uniq_tracks==-1)
    for i in range(uniq_tracks.shape[0]):
        # Delete occasional tracks
        if counts[i] < (num_frames/10.0):
            idx = torch.nonzero(uniq_tracks[i]==trackids,as_tuple=True)[0]
            trackids[idx] = -1
            counts[dumpster] += counts[i]
            uniq_tracks[i] = -1
            counts[i] = 0

    for i in range(uniq_tracks.shape[0]):
        # Merge by simple disjoint set union'ish algorithm. Prove that this would always converge before N^2 steps?
        if not uniq_tracks[i]==-1:
            min_delay_after, min_delay_before = 100000, 100000
            arrid_after, arrid_before = -1, -1
            idx1 = torch.nonzero(uniq_tracks[i]==trackids,as_tuple=True)[0]
            frame1 = frames[idx1]
            start1, end1 = frame1.min().item(), frame1.max().item()
            
            for j in range(uniq_tracks.shape[0]):
                if uniq_tracks[j]!=-1 and j!=i:
                    idx2 = torch.nonzero(uniq_tracks[j]==trackids,as_tuple=True)[0]
                    isoverlap = np.intersect1d(idx1.numpy(), idx2.numpy()).shape[0]
                    if not isoverlap:
                        frame2 = frames[idx2] 
                        start2, end2 = frame2.min().item(), frame2.max().item()
                        if start2 > end1 and (start2-end1)<min_delay_after:
                            min_delay_after = start2-end1
                            arrid_after = j
                        elif end2 > start1 and end2-start1<min_delay_before:
                            min_delay_before = end2-start1
                            arrid_before = j

            if arrid_after != -1:
                idx2 = torch.nonzero(uniq_tracks[arrid_after]==trackids,as_tuple=True)[0]
                trackids[idx2] = uniq_tracks[i]
                counts[i] += counts[arrid_after]
                uniq_tracks[arrid_after] = -1
                counts[arrid_after] = 0
            if arrid_before != -1:
                idx2 = torch.nonzero(uniq_tracks[arrid_before]==trackids,as_tuple=True)[0]
                trackids[idx2] = uniq_tracks[i]
                counts[i] += counts[arrid_before]
                uniq_tracks[arrid_before] = -1
                counts[arrid_before] = 0
    uniq_tracks, counts = torch.unique(trackids, return_counts=True)
    
    tracked_detections = []
    for track in uniq_tracks:
        if track !=-1:
            idx = torch.nonzero(track==trackids,as_tuple=True)[0]
            tracklet_bboxes = bboxes[idx]
            tracklet_landmarks = landmarks[idx]
            tracklet_confidence = confidence[idx]
            tracklet_frames = frames[idx]
            tracklet_smooth_bboxes = smooth_bboxes[idx]
            tracked_detections.append((tracklet_bboxes,tracklet_landmarks,tracklet_confidence, tracklet_frames,tracklet_smooth_bboxes))

    return tracked_detections
