function objects = findAllOjbects(label_dir, nimages)
objects = containers.Map;
% find all the objects in the whole sequence
for img_idx = 1:nimages
    object = readLabels(label_dir, img_idx-1);
    if ~isempty(object)
        for obj_idx = 1:numel(object)
            objects(object(obj_idx).ID_name) = object(obj_idx);
        end
    end
    
end