% cmd: ls -lR |grep -v ^d|awk '{print $9}' | sed "s:^:`pwd`/:" >> input16.txt
% find -type f -name "C201*" |wc -l

fid_in = fopen('./input16.txt');
fid_out = fopen('./output16.txt', 'at');

while ~feof(fid_in)
    str = fgetl(fid_in);
    load(str);

    in_file = map(1000);
    fprintf(fid_out,'%0.4f', in_file);
    fprintf(fid_out,'\n');
end

fclose(fid_in);
fclose(fid_out);