function varargout = subsref(self, S) 
   if length(S) == 1 
      if S(1).type == '.'
         switch S(1).subs
            case 'shape'
               varargout = {size(self.wts)};
            case 'num_hid'
               varargout = {size(self.wts, 2)};
            case 'num_dims'
               varargout = {size(self.wts, 1)};
            otherwise
               varargout = {builtin('subsref', self, S)};
         end 
         return ; 
      elseif S(1).type == '()'
         varargout = {builtin('subsref', self, S)};
         return
      else
        error('Unsupported reference. Extend or mend your ways');
      end
  end

  if length(S) == 2
    switch S(1).type
       case '()'
          varargout = {subsref(self(S(1).subs{1}), S(2))};
          return
       case '.'
          switch S(1).subs
             case 'apply_gradients'
                varargout = {apply_gradients(self, S(2).subs{1}, ...
                                                 S(2).subs{2}, S(2).subs{3})};
                return
             case 'fwd_prop'
                varargout = {fwd_prop(self, S(2).subs{1})};
                return
             case 'compute_accuracy'
                varargout = {compute_accuracy(self, S(2).subs{1}, ...
                                                              S(2).subs{2})};
                return
             case 'compute_act_gradients_from_targets'
                varargout = {compute_act_gradients_from_targets(self, ...
                                                S(2).subs{1}, S(2).subs{2})};
                return
             case 'back_prop'
                varargout = {back_prop(self, S(2).subs{1}, S(2).subs{2})};
                return
             case 'compute_act_grad_from_output_grad'
                varargout = {compute_act_grad_from_output_grad(self, ...
                                                S(2).subs{1}, S(2).subs{2})};
                return
             otherwise
                varargout = {builtin('subsref', self, S)};
                return
          end
       otherwise
          S(1)
          S(2)
          error('Unsupported reference. Extend or mend your ways');
    end 
 end

 if length(S) == 3
    switch S(1).type
       case '()'
          varargout = {subsref(self(S(1).subs{1}), S(2:end))};
          return
       otherwise
          error('Unsupported reference of length 3. Extend or mend your ways');
    end
 end

 error('Unsupported reference. Extend or mend your ways');
end
