# -*- coding:utf-8 -*-

import os
import time
import s3fs

class S3FileSystemPatched(s3fs.S3FileSystem):
    def __init__(self, *k, **kw):
        super(S3FileSystemPatched, self).__init__(*k,
                                                  key=os.environ['AWS_ACCESS_KEY_ID'],
                                                  secret=os.environ['AWS_SECRET_ACCESS_KEY'],
                                                  client_kwargs={'endpoint_url': 'http://' + os.environ['S3_ENDPOINT']},
                                                  # client_kwargs={'endpoint_url': '' + os.environ['S3_ENDPOINT']},
                                                  **kw
                                                  )


class S3Filewrite:
    def __init__(self, args):
        super(S3Filewrite, self).__init__()
        self.output_path = args.data_output

    def write(self, pred, label, src_id, dst_id, file_idx):
        s3fs.S3FileSystem = S3FileSystemPatched
        fs = s3fs.S3FileSystem()
        start = time.time()

        with fs.open(self.output_path + 'pred_{}.csv'.format(file_idx), mode="w") as resultfile:
            # data = [line.decode('utf8').strip() for line in data.tolist()]
            for i, p in enumerate(data):

                line = "{},{},{},{}\n".format(src_id[i][0], dst_id[i][0], p[i][0], p[i][1])
                resultfile.write(line)
                if i%1000==0:
                    print("write:",i)

        cost = time.time() - start
        print("wrote {} lines with {:.2f}s".format(len(data), cost))
