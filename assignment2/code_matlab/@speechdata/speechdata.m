function self = speechdata(file_name, num_frames_per_pt)
   obj = load_data(file_name, num_frames_per_pt);
   self = class(obj, 'speechdata');
end

function obj = load_data(file_name, num_frames_per_pt)
   d = builtin('load', file_name);
   obj = struct();
   obj.num_frames_per_pt = num_frames_per_pt;
   obj.data = d.data; 

   try
      obj.targets = d.targets;
   catch
      fprintf('targets not specified in file: %s\n',file_name)
   end
   obj.utt_indices = d.utt_indices+1;
   obj.label_dim = d.label_dim;
   obj.label_eye = eye(d.label_dim);
   obj.num_utterances = size(d.utt_indices,1);
   obj.frame_dim = size(d.data,1);
   obj.data_dim = obj.frame_dim*obj.num_frames_per_pt;
   obj.num_pts = 0;
   obj.lst_indices = {};
   for i=1:obj.num_utterances
      s = obj.utt_indices(i,1);
      e = obj.utt_indices(i,2);
      indices = s:(e-obj.num_frames_per_pt);
      obj.lst_indices = {obj.lst_indices{:},indices};
      obj.num_pts = obj.num_pts + size(indices,2);
   end
   
   obj.indices = zeros(1,obj.num_pts);
   num_pts_so_far = 0;
   for indices=obj.lst_indices
      indices = indices{1};
      obj.indices(num_pts_so_far+1:num_pts_so_far+1+size(indices,2)-1) = indices;
      num_pts_so_far = num_pts_so_far + size(indices,2);
   end

   fprintf('Loaded %d points\n',num_pts_so_far)
   obj.data_mean = mean(obj.data,2);
   obj.data_std = std(obj.data,[],2);
end
