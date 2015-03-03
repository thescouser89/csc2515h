function varargout = subsref(self, S)
   if length(S) == 1 
      if S(1).type == '.'
         varargout = {builtin('subsref', self, S)};
         return ; 
      elseif S(1).type == '()'
         varargout = {builtin('subsref', self, S)};
         return
      else
        error('Unsupported reference. Extend or mend your ways');
      end
  end

   if length(S) == 2
      if strcmp(S(1).subs, 'save') == 1
         varargout = {};
         save(self, S(2).subs{1});
         return
      elseif strcmp(S(1).subs, 'load') == 1
         varargout = {load(self, S(2).subs{1})};
         return
      elseif strcmp(S(1).subs, 'test') == 1
         varargout = {test(self, S(2).subs{1})};
         return
      elseif strcmp(S(1).subs, 'create_predictions') == 1
         varargout = {create_predictions(self, S(2).subs{1})};
         return
      elseif strcmp(S(1).subs, 'apply_gradients') == 1
         varargout = {apply_gradients(self, S(2).subs{1}, S(2).subs{2}, ...
                                        S(2).subs{3})};
         return
      elseif strcmp(S(1).subs, 'train_for_one_epoch') == 1
         varargout = {train_for_one_epoch(self, S(2).subs{1}, S(2).subs{2}, ...
                                          S(2).subs{3}, S(2).subs{4}, ...
                                          S(2).subs{5})};
         return
      elseif strcmp(S(1).subs, 'fwd_prop') == 1
         varargout = fwd_prop(self, S(2).subs{1});
         varargout
         return
      elseif strcmp(S(1).subs, 'back_prop') == 1
         varargout = {back_prop(self, S(2).subs{1}, S(2).subs{2})};
         return
      elseif strcmp(S(1).subs, 'lst_layer_type') == 1
         if length(S(2).subs) == 1
            varargout = {self.lst_layer_type{S(2).subs{1}}};
         else
            varargout = {self.lst_layer_type{S(2).subs}};
         end
         return
      elseif strcmp(S(1).subs, 'lst_layers') == 1
         if length(S(2).subs) == 1
            varargout = {self.lst_layers(S(2).subs{1})};
         else
            varargout = {self.lst_layers(S(2).subs)};
         end
         return
      elseif strcmp(S(1).subs, 'lst_num_hid') == 1
         if length(S(2).subs) == 1
            varargout = {self.lst_num_hid(S(2).subs{1})};
         else
            varargout = {self.lst_num_hid(S(2).subs)};
         end
         return
      else 
        fprintf (1,'Unsupported S(1).subs = ')
        S(1).subs

      end
   end

   if length(S) > 2
      fprintf(1, 'here\n');
    switch S(1).type
       case '()'
          varargout = subsref(self(S(1).subs{1}), S(2:end));
          return
       case '.'
          varargout = subsref(self, S(2:end));
          return
       otherwise
          error('Unsupported reference of length 3. Extend or mend your ways');
    end
 end
end
