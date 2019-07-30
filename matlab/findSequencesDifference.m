function [objects_not_in_A, objects_not_in_B, objects_in_common] = ...
        findSequencesDifference(seq_GT_dir_A, seq_GT_dir_B, plot_flag)
% this function computes the object differences between two sequences and
% can also show the difference
% output: 
%       object_not_in_A: a struct array that contains objects in seq B but
%                        not in seq A
%       object_not_in_B: a struct array that contains objects in seq A but
%                        not in seq B
%       object_in_common: a struct array that contains objects both in seq
%                         A and seq B
% input:
%       seq_GT_dir_A: the label dir of seq A
%       seq_GT_dir_B: the label dir of seq B
%       plot_flag: bool, true - plot the difference, false - do not plot


% - 1 substract labels.json in label dir
num_labels_A = length(dir(fullfile(seq_GT_dir_A, '*.json'))) - 1;
num_labels_B = length(dir(fullfile(seq_GT_dir_B, '*.json'))) - 1;

% find all objects in sequence A and sequence B
all_objects_A = findAllOjbects(seq_GT_dir_A, num_labels_A);
all_objects_B = findAllOjbects(seq_GT_dir_B, num_labels_B);

object_keys_A = keys(all_objects_A);
object_keys_B = keys(all_objects_B);

objects_in_common = struct([]);
% find objects that is in sequence A but not in sequence B
objects_not_in_B = struct([]);
for key_idx_A = 1:numel(object_keys_A)
    current_key_A = object_keys_A{key_idx_A};
    if ~ismember(current_key_A, object_keys_B)
        current_object_A = all_objects_A(current_key_A);
        % compute 3D cube origin and extent for plotting purpose
        [box_origin_A, box_extent_A] = compute3DBoxCorners(current_object_A);
        current_object_A.cube_origin = box_origin_A;
        current_object_A.cube_extent = box_extent_A;
        objects_not_in_B = [objects_not_in_B, current_object_A];
    else
        current_object_A = all_objects_A(current_key_A);
        % compute 3D cube origin and extent for plotting purpose
        [box_origin_A, box_extent_A] = compute3DBoxCorners(current_object_A);
        current_object_A.cube_origin = box_origin_A;
        current_object_A.cube_extent = box_extent_A;
        objects_in_common = [objects_in_common, current_object_A];
    end
end

% find objects that is in sequence A but not in sequence B
objects_not_in_A = struct([]);
for key_idx_B = 1:numel(object_keys_B)
    current_key_B = object_keys_B{key_idx_B};
    if ~ismember(current_key_B, object_keys_A)
        current_object_B = all_objects_B(current_key_B);
        % compute 3D cube origin and extent for plotting purpose
        [box_origin_B, box_extent_B] = compute3DBoxCorners(current_object_B);
        current_object_B.cube_origin = box_origin_B;
        current_object_B.cube_extent = box_extent_B;
        objects_not_in_A = [objects_not_in_A, current_object_B];
    end
end

if plot_flag
    figure(100); h = subplot(1,1,1);axis equal;
    hold(h, 'on');
    % plot objects in common in green 
    for idx = 1:numel(objects_in_common)
        current_object = objects_in_common(idx);
        hh = plotcube(current_object.cube_extent/100.0, current_object.cube_origin/100.0, .1, [0, 1, 0], h);
    end
    % plot objects not in sequence A in blue
    for idx = 1:numel(objects_not_in_A)
        current_object = objects_not_in_A(idx);
        hh = plotcube(current_object.cube_extent/100.0, current_object.cube_origin/100.0, .1, [0, 0, 1], h);
        text(current_object.t(1)/100, -current_object.t(2)/100, current_object.t(3)/100 + 1, ...
             current_object.type,'HorizontalAlignment','center','FontSize',8, 'color','b');
    end
    % plot objects not in sequence B in red
    for idx = 1:numel(objects_not_in_B)
        current_object = objects_not_in_B(idx);
        hh = plotcube(current_object.cube_extent/100.0, current_object.cube_origin/100.0, .1, [1, 0, 0], h);
        text(current_object.t(1)/100, -current_object.t(2)/100, current_object.t(3)/100 + 1, ...
             current_object.type,'HorizontalAlignment','center','FontSize',8, 'color','r');
    end
    title('The difference between two sequences: \color{red}red - removed, \color{blue}blue - added');
    
end

