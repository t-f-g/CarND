cmake .. -G "Unix Makefiles" && make
.\UnscentedKF.exe ..\data\sample-laser-radar-measurement-data-1.txt slrmd1.txt
.\UnscentedKF.exe ..\data\sample-laser-radar-measurement-data-2.txt slrmd2.txt
.\UnscentedKF.exe ..\data\obj_pose-laser-radar-synthetic-input.txt oplrsi.txt