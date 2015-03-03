function self = randomize_data_indices(self)
   p = randperm(numel(self.indices));
   indices = self.indices;
   self.indices = indices(p);
