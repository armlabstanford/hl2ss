%%
% This script receives video from the HoloLens depth camera in long throw
% mode and plays it.
% Close the figure to stop.

%% Settings

% HoloLens address
host = '192.169.1.41';

%%

client = hl2ss.mt.sink_rm_depth_longthrow(host, hl2ss.stream_port.RM_DEPTH_LONGTHROW);
calibration = client.download_calibration();
client.open();

h = []; % figure handle

try
while (true)
    data = client.get_packet_by_index(-1); % -1 for most recent frame
    if (data.status == 0) % got packet
        % normalize for visibility
        depth = double(data.depth);
        ab = double(data.ab);
        depth = depth / max(depth, [], 'all');
        ab = ab / max(ab, [], 'all');
        frame = [depth, ab] * 255;
        if (isempty(h))
            h = image(frame);
            colormap gray
        else
            h.CData = frame;
        end
        drawnow
    else % no data
        pause(1); % wait for data
    end
end
catch ME
    disp(ME.message);
end

client.close();
