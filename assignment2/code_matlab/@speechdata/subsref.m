function varargout = subsref(self, S) 
  if length(S) == 1 & S(1).type == '.'
     varargout = {builtin('subsref', self, S)};
     return 
  end

  if length(S) == 2
     if strcmp(S(1).type, '.') == 1
        if strcmp(S(1).subs, 'get_utterance_data') == 1
           varargout = {get_utterance_data(self, S(2).subs{1}, S(2).subs{2})};
           return
        elseif strcmp(S(1).subs, 'indices') == 1
           varargout = {builtin('subsref', self.indices, S(2))};
           return
        elseif strcmp(S(1).subs, 'get_normalized_data') == 1
           varargout = {get_normalized_data(self)};
           return
        elseif strcmp(S(1).subs, 'get_randomized_data') == 1
           varargout = cell(1,2);
           [varargout{:}] = get_randomized_data(self, S(2).subs{1}, S(2).subs{2});
           return
        elseif strcmp(S(1).subs, 'randomize_data_indices') == 1
           varargout = {randomize_data_indices(self)};
           return
        elseif strcmp(S(1).subs, 'get_data_dim') == 1
           varargout = {get_data_dim(self)};
           return
        elseif strcmp(S(1).subs, 'get_num_utterances') == 1
           varargout = {get_num_utterances(self)};
           return
        elseif strcmp(S(1).subs, 'get_target_dim') == 1
           varargout = {get_target_dim(self)};
           return
        end
     end
  end
end
