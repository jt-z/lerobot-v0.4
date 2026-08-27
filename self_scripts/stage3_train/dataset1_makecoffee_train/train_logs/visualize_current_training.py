import matplotlib.pyplot as plt
import re
import numpy as np

# Training log data - your complete log
log_data = """INFO 2026-08-10 17:20:55 ot_train.py:423 step:39K smpl:20M ep:22K epch:555.39 loss:0.032 grdn:0.663 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-10 17:22:04 ot_train.py:423 step:39K smpl:20M ep:22K epch:558.25 loss:0.033 grdn:0.645 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 17:23:15 ot_train.py:423 step:39K smpl:20M ep:22K epch:561.11 loss:0.032 grdn:0.669 lr:5.0e-05 updt_s:0.325 data_s:0.025
INFO 2026-08-10 17:24:24 ot_train.py:423 step:39K smpl:20M ep:23K epch:563.97 loss:0.032 grdn:inf lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 17:25:33 ot_train.py:423 step:40K smpl:20M ep:23K epch:566.84 loss:0.033 grdn:0.674 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 17:26:44 ot_train.py:423 step:40K smpl:20M ep:23K epch:569.70 loss:0.033 grdn:0.704 lr:5.0e-05 updt_s:0.330 data_s:0.026
INFO 2026-08-10 17:27:53 ot_train.py:423 step:40K smpl:20M ep:23K epch:572.56 loss:0.032 grdn:0.679 lr:5.0e-05 updt_s:0.323 data_s:0.019
INFO 2026-08-10 17:27:53 ot_train.py:443 Checkpoint policy after step 40000
INFO 2026-08-10 17:29:03 ot_train.py:423 step:40K smpl:21M ep:23K epch:575.43 loss:0.032 grdn:0.675 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-10 17:30:13 ot_train.py:423 step:40K smpl:21M ep:23K epch:578.29 loss:0.032 grdn:0.662 lr:5.0e-05 updt_s:0.327 data_s:0.024
INFO 2026-08-10 17:31:23 ot_train.py:423 step:41K smpl:21M ep:23K epch:581.15 loss:0.033 grdn:0.645 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 17:32:32 ot_train.py:423 step:41K smpl:21M ep:23K epch:584.01 loss:0.032 grdn:0.656 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 17:33:42 ot_train.py:423 step:41K smpl:21M ep:23K epch:586.88 loss:0.032 grdn:0.606 lr:5.0e-05 updt_s:0.322 data_s:0.027
INFO 2026-08-10 17:34:51 ot_train.py:423 step:41K smpl:21M ep:24K epch:589.74 loss:0.032 grdn:0.649 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 17:36:02 ot_train.py:423 step:41K smpl:21M ep:24K epch:592.60 loss:0.031 grdn:inf lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-10 17:37:12 ot_train.py:423 step:42K smpl:21M ep:24K epch:595.47 loss:0.032 grdn:0.636 lr:5.0e-05 updt_s:0.330 data_s:0.019
INFO 2026-08-10 17:38:21 ot_train.py:423 step:42K smpl:21M ep:24K epch:598.33 loss:0.031 grdn:0.638 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-10 17:39:31 ot_train.py:423 step:42K smpl:22M ep:24K epch:601.19 loss:0.031 grdn:0.647 lr:5.0e-05 updt_s:0.328 data_s:0.025
INFO 2026-08-10 17:39:31 ot_train.py:443 Checkpoint policy after step 42000
INFO 2026-08-10 17:40:41 ot_train.py:423 step:42K smpl:22M ep:24K epch:604.05 loss:0.031 grdn:0.591 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-10 17:41:51 ot_train.py:423 step:42K smpl:22M ep:24K epch:606.92 loss:0.032 grdn:0.665 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 17:43:01 ot_train.py:423 step:43K smpl:22M ep:24K epch:609.78 loss:0.031 grdn:0.642 lr:5.0e-05 updt_s:0.324 data_s:0.025
INFO 2026-08-10 17:44:10 ot_train.py:423 step:43K smpl:22M ep:25K epch:612.64 loss:0.031 grdn:0.594 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-10 17:45:20 ot_train.py:423 step:43K smpl:22M ep:25K epch:615.51 loss:0.031 grdn:0.641 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 17:46:30 ot_train.py:423 step:43K smpl:22M ep:25K epch:618.37 loss:0.031 grdn:0.612 lr:5.0e-05 updt_s:0.324 data_s:0.025
INFO 2026-08-10 17:47:39 ot_train.py:423 step:43K smpl:22M ep:25K epch:621.23 loss:0.030 grdn:inf lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 17:48:48 ot_train.py:423 step:44K smpl:22M ep:25K epch:624.09 loss:0.030 grdn:0.636 lr:5.0e-05 updt_s:0.323 data_s:0.019
INFO 2026-08-10 17:49:59 ot_train.py:423 step:44K smpl:22M ep:25K epch:626.96 loss:0.030 grdn:0.677 lr:5.0e-05 updt_s:0.330 data_s:0.024
INFO 2026-08-10 17:51:08 ot_train.py:423 step:44K smpl:23M ep:25K epch:629.82 loss:0.031 grdn:0.611 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-10 17:51:08 ot_train.py:443 Checkpoint policy after step 44000
INFO 2026-08-10 17:52:20 ot_train.py:423 step:44K smpl:23M ep:25K epch:632.68 loss:0.031 grdn:0.626 lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-10 17:53:29 ot_train.py:423 step:44K smpl:23M ep:25K epch:635.54 loss:0.030 grdn:0.597 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 17:54:38 ot_train.py:423 step:45K smpl:23M ep:26K epch:638.41 loss:0.030 grdn:0.628 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 17:55:48 ot_train.py:423 step:45K smpl:23M ep:26K epch:641.27 loss:0.030 grdn:0.685 lr:5.0e-05 updt_s:0.327 data_s:0.025
INFO 2026-08-10 17:56:58 ot_train.py:423 step:45K smpl:23M ep:26K epch:644.13 loss:0.029 grdn:0.614 lr:5.0e-05 updt_s:0.329 data_s:0.019
INFO 2026-08-10 17:58:07 ot_train.py:423 step:45K smpl:23M ep:26K epch:647.00 loss:0.030 grdn:0.564 lr:5.0e-05 updt_s:0.322 data_s:0.019
INFO 2026-08-10 17:59:18 ot_train.py:423 step:45K smpl:23M ep:26K epch:649.86 loss:0.031 grdn:inf lr:5.0e-05 updt_s:0.329 data_s:0.026
INFO 2026-08-10 18:00:27 ot_train.py:423 step:46K smpl:23M ep:26K epch:652.72 loss:0.030 grdn:0.639 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 18:01:36 ot_train.py:423 step:46K smpl:23M ep:26K epch:655.58 loss:0.030 grdn:0.617 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-10 18:02:47 ot_train.py:423 step:46K smpl:24M ep:26K epch:658.45 loss:0.030 grdn:0.644 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-10 18:02:47 ot_train.py:443 Checkpoint policy after step 46000
INFO 2026-08-10 18:03:57 ot_train.py:423 step:46K smpl:24M ep:26K epch:661.31 loss:0.030 grdn:0.596 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-10 18:05:08 ot_train.py:423 step:46K smpl:24M ep:27K epch:664.17 loss:0.030 grdn:0.621 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-10 18:06:17 ot_train.py:423 step:47K smpl:24M ep:27K epch:667.04 loss:0.030 grdn:0.610 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 18:07:26 ot_train.py:423 step:47K smpl:24M ep:27K epch:669.90 loss:0.029 grdn:0.590 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 18:08:37 ot_train.py:423 step:47K smpl:24M ep:27K epch:672.76 loss:0.030 grdn:0.595 lr:5.0e-05 updt_s:0.329 data_s:0.025
INFO 2026-08-10 18:09:46 ot_train.py:423 step:47K smpl:24M ep:27K epch:675.62 loss:0.029 grdn:0.602 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 18:10:55 ot_train.py:423 step:47K smpl:24M ep:27K epch:678.49 loss:0.030 grdn:inf lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 18:12:06 ot_train.py:423 step:48K smpl:24M ep:27K epch:681.35 loss:0.029 grdn:0.626 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-10 18:13:15 ot_train.py:423 step:48K smpl:24M ep:27K epch:684.21 loss:0.029 grdn:0.590 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 18:14:24 ot_train.py:423 step:48K smpl:25M ep:27K epch:687.08 loss:0.029 grdn:0.602 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 18:14:24 ot_train.py:443 Checkpoint policy after step 48000
INFO 2026-08-10 18:15:36 ot_train.py:423 step:48K smpl:25M ep:28K epch:689.94 loss:0.029 grdn:0.615 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-10 18:16:45 ot_train.py:423 step:48K smpl:25M ep:28K epch:692.80 loss:0.029 grdn:0.596 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 18:17:54 ot_train.py:423 step:49K smpl:25M ep:28K epch:695.66 loss:0.029 grdn:0.626 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 18:19:05 ot_train.py:423 step:49K smpl:25M ep:28K epch:698.53 loss:0.029 grdn:0.646 lr:5.0e-05 updt_s:0.330 data_s:0.025
INFO 2026-08-10 18:20:14 ot_train.py:423 step:49K smpl:25M ep:28K epch:701.39 loss:0.029 grdn:0.635 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 18:21:25 ot_train.py:423 step:49K smpl:25M ep:28K epch:704.25 loss:0.029 grdn:0.608 lr:5.0e-05 updt_s:0.328 data_s:0.025
INFO 2026-08-10 18:22:34 ot_train.py:423 step:49K smpl:25M ep:28K epch:707.12 loss:0.029 grdn:inf lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 18:23:44 ot_train.py:423 step:50K smpl:25M ep:28K epch:709.98 loss:0.029 grdn:0.623 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-10 18:24:54 ot_train.py:423 step:50K smpl:25M ep:29K epch:712.84 loss:0.028 grdn:0.605 lr:5.0e-05 updt_s:0.326 data_s:0.025
INFO 2026-08-10 18:26:04 ot_train.py:423 step:50K smpl:26M ep:29K epch:715.70 loss:0.029 grdn:0.626 lr:5.0e-05 updt_s:0.327 data_s:0.020
INFO 2026-08-10 18:26:04 ot_train.py:443 Checkpoint policy after step 50000
INFO 2026-08-10 18:27:14 ot_train.py:423 step:50K smpl:26M ep:29K epch:718.57 loss:0.029 grdn:0.597 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 18:28:24 ot_train.py:423 step:50K smpl:26M ep:29K epch:721.43 loss:0.028 grdn:0.593 lr:5.0e-05 updt_s:0.325 data_s:0.026
INFO 2026-08-10 18:29:33 ot_train.py:423 step:51K smpl:26M ep:29K epch:724.29 loss:0.029 grdn:0.627 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 18:30:43 ot_train.py:423 step:51K smpl:26M ep:29K epch:727.15 loss:0.029 grdn:0.617 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 18:31:54 ot_train.py:423 step:51K smpl:26M ep:29K epch:730.02 loss:0.028 grdn:0.557 lr:5.0e-05 updt_s:0.329 data_s:0.026
INFO 2026-08-10 18:33:02 ot_train.py:423 step:51K smpl:26M ep:29K epch:732.88 loss:0.029 grdn:0.592 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-10 18:34:12 ot_train.py:423 step:51K smpl:26M ep:29K epch:735.74 loss:0.028 grdn:inf lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 18:35:22 ot_train.py:423 step:52K smpl:26M ep:30K epch:738.61 loss:0.029 grdn:0.601 lr:5.0e-05 updt_s:0.328 data_s:0.025
INFO 2026-08-10 18:36:31 ot_train.py:423 step:52K smpl:27M ep:30K epch:741.47 loss:0.028 grdn:0.583 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 18:37:41 ot_train.py:423 step:52K smpl:27M ep:30K epch:744.33 loss:0.029 grdn:0.613 lr:5.0e-05 updt_s:0.326 data_s:0.024
INFO 2026-08-10 18:37:41 ot_train.py:443 Checkpoint policy after step 52000
INFO 2026-08-10 18:38:52 ot_train.py:423 step:52K smpl:27M ep:30K epch:747.19 loss:0.029 grdn:0.590 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-10 18:40:00 ot_train.py:423 step:52K smpl:27M ep:30K epch:750.06 loss:0.028 grdn:0.594 lr:5.0e-05 updt_s:0.322 data_s:0.019
INFO 2026-08-10 18:41:11 ot_train.py:423 step:53K smpl:27M ep:30K epch:752.92 loss:0.027 grdn:0.561 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-10 18:42:20 ot_train.py:423 step:53K smpl:27M ep:30K epch:755.78 loss:0.028 grdn:0.614 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 18:43:29 ot_train.py:423 step:53K smpl:27M ep:30K epch:758.65 loss:0.028 grdn:0.615 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-10 18:44:40 ot_train.py:423 step:53K smpl:27M ep:30K epch:761.51 loss:0.028 grdn:0.607 lr:5.0e-05 updt_s:0.329 data_s:0.025
INFO 2026-08-10 18:45:49 ot_train.py:423 step:53K smpl:27M ep:31K epch:764.37 loss:0.029 grdn:inf lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 18:46:58 ot_train.py:423 step:54K smpl:27M ep:31K epch:767.23 loss:0.028 grdn:0.573 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 18:48:09 ot_train.py:423 step:54K smpl:28M ep:31K epch:770.10 loss:0.028 grdn:0.615 lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-10 18:49:19 ot_train.py:423 step:54K smpl:28M ep:31K epch:772.96 loss:0.028 grdn:0.585 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-10 18:49:19 ot_train.py:443 Checkpoint policy after step 54000
INFO 2026-08-10 18:50:28 ot_train.py:423 step:54K smpl:28M ep:31K epch:775.82 loss:0.028 grdn:0.572 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 18:51:39 ot_train.py:423 step:54K smpl:28M ep:31K epch:778.69 loss:0.028 grdn:0.589 lr:5.0e-05 updt_s:0.329 data_s:0.026
INFO 2026-08-10 18:52:48 ot_train.py:423 step:55K smpl:28M ep:31K epch:781.55 loss:0.028 grdn:0.587 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 18:53:59 ot_train.py:423 step:55K smpl:28M ep:31K epch:784.41 loss:0.028 grdn:0.568 lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-10 18:55:08 ot_train.py:423 step:55K smpl:28M ep:31K epch:787.27 loss:0.028 grdn:0.585 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 18:56:18 ot_train.py:423 step:55K smpl:28M ep:32K epch:790.14 loss:0.027 grdn:0.604 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 18:57:28 ot_train.py:423 step:55K smpl:28M ep:32K epch:793.00 loss:0.027 grdn:inf lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-10 18:58:37 ot_train.py:423 step:56K smpl:28M ep:32K epch:795.86 loss:0.028 grdn:0.611 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-10 18:59:46 ot_train.py:423 step:56K smpl:29M ep:32K epch:798.73 loss:0.028 grdn:0.585 lr:5.0e-05 updt_s:0.323 data_s:0.019
INFO 2026-08-10 19:00:57 ot_train.py:423 step:56K smpl:29M ep:32K epch:801.59 loss:0.027 grdn:0.606 lr:5.0e-05 updt_s:0.330 data_s:0.024
INFO 2026-08-10 19:00:57 ot_train.py:443 Checkpoint policy after step 56000
INFO 2026-08-10 19:02:06 ot_train.py:423 step:56K smpl:29M ep:32K epch:804.45 loss:0.027 grdn:0.611 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 19:03:16 ot_train.py:423 step:56K smpl:29M ep:32K epch:807.31 loss:0.027 grdn:0.596 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 19:04:27 ot_train.py:423 step:57K smpl:29M ep:32K epch:810.18 loss:0.027 grdn:0.573 lr:5.0e-05 updt_s:0.331 data_s:0.025
INFO 2026-08-10 19:05:36 ot_train.py:423 step:57K smpl:29M ep:33K epch:813.04 loss:0.027 grdn:0.597 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-10 19:06:45 ot_train.py:423 step:57K smpl:29M ep:33K epch:815.90 loss:0.027 grdn:0.569 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 19:07:56 ot_train.py:423 step:57K smpl:29M ep:33K epch:818.76 loss:0.027 grdn:0.588 lr:5.0e-05 updt_s:0.330 data_s:0.024
INFO 2026-08-10 19:09:05 ot_train.py:423 step:57K smpl:29M ep:33K epch:821.63 loss:0.027 grdn:0.607 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 19:10:16 ot_train.py:423 step:58K smpl:29M ep:33K epch:824.49 loss:0.027 grdn:inf lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-10 19:11:25 ot_train.py:423 step:58K smpl:30M ep:33K epch:827.35 loss:0.027 grdn:0.598 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 19:12:34 ot_train.py:423 step:58K smpl:30M ep:33K epch:830.22 loss:0.027 grdn:0.548 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 19:12:34 ot_train.py:443 Checkpoint policy after step 58000
INFO 2026-08-10 19:13:45 ot_train.py:423 step:58K smpl:30M ep:33K epch:833.08 loss:0.027 grdn:0.559 lr:5.0e-05 updt_s:0.327 data_s:0.024
INFO 2026-08-10 19:14:55 ot_train.py:423 step:58K smpl:30M ep:33K epch:835.94 loss:0.027 grdn:0.556 lr:5.0e-05 updt_s:0.330 data_s:0.019
INFO 2026-08-10 19:16:04 ot_train.py:423 step:59K smpl:30M ep:34K epch:838.80 loss:0.027 grdn:0.613 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 19:17:15 ot_train.py:423 step:59K smpl:30M ep:34K epch:841.67 loss:0.026 grdn:0.566 lr:5.0e-05 updt_s:0.329 data_s:0.026
INFO 2026-08-10 19:18:24 ot_train.py:423 step:59K smpl:30M ep:34K epch:844.53 loss:0.027 grdn:0.579 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 19:19:33 ot_train.py:423 step:59K smpl:30M ep:34K epch:847.39 loss:0.027 grdn:0.580 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 19:20:44 ot_train.py:423 step:59K smpl:30M ep:34K epch:850.26 loss:0.026 grdn:0.562 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-10 19:21:53 ot_train.py:423 step:60K smpl:31M ep:34K epch:853.12 loss:0.026 grdn:inf lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 19:23:02 ot_train.py:423 step:60K smpl:31M ep:34K epch:855.98 loss:0.027 grdn:0.595 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 19:24:13 ot_train.py:423 step:60K smpl:31M ep:34K epch:858.84 loss:0.027 grdn:0.576 lr:5.0e-05 updt_s:0.328 data_s:0.025
INFO 2026-08-10 19:24:13 ot_train.py:443 Checkpoint policy after step 60000
INFO 2026-08-10 19:25:23 ot_train.py:423 step:60K smpl:31M ep:34K epch:861.71 loss:0.027 grdn:0.571 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 19:26:33 ot_train.py:423 step:60K smpl:31M ep:35K epch:864.57 loss:0.027 grdn:0.578 lr:5.0e-05 updt_s:0.325 data_s:0.025
INFO 2026-08-10 19:27:43 ot_train.py:423 step:61K smpl:31M ep:35K epch:867.43 loss:0.027 grdn:0.565 lr:5.0e-05 updt_s:0.329 data_s:0.019
INFO 2026-08-10 19:28:51 ot_train.py:423 step:61K smpl:31M ep:35K epch:870.30 loss:0.027 grdn:0.579 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-10 19:30:02 ot_train.py:423 step:61K smpl:31M ep:35K epch:873.16 loss:0.026 grdn:0.559 lr:5.0e-05 updt_s:0.330 data_s:0.024
INFO 2026-08-10 19:31:12 ot_train.py:423 step:61K smpl:31M ep:35K epch:876.02 loss:0.026 grdn:0.580 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 19:32:21 ot_train.py:423 step:61K smpl:31M ep:35K epch:878.88 loss:0.025 grdn:0.550 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 19:33:32 ot_train.py:423 step:62K smpl:32M ep:35K epch:881.75 loss:0.026 grdn:inf lr:5.0e-05 updt_s:0.328 data_s:0.025
INFO 2026-08-10 19:34:41 ot_train.py:423 step:62K smpl:32M ep:35K epch:884.61 loss:0.026 grdn:0.567 lr:5.0e-05 updt_s:0.323 data_s:0.019
INFO 2026-08-10 19:35:50 ot_train.py:423 step:62K smpl:32M ep:35K epch:887.47 loss:0.026 grdn:0.563 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 19:35:50 ot_train.py:443 Checkpoint policy after step 62000
INFO 2026-08-10 19:37:01 ot_train.py:423 step:62K smpl:32M ep:36K epch:890.34 loss:0.026 grdn:0.558 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-10 19:38:11 ot_train.py:423 step:62K smpl:32M ep:36K epch:893.20 loss:0.026 grdn:0.584 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-10 19:39:20 ot_train.py:423 step:63K smpl:32M ep:36K epch:896.06 loss:0.026 grdn:0.550 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 19:40:30 ot_train.py:423 step:63K smpl:32M ep:36K epch:898.92 loss:0.026 grdn:0.565 lr:5.0e-05 updt_s:0.327 data_s:0.025
INFO 2026-08-10 19:41:40 ot_train.py:423 step:63K smpl:32M ep:36K epch:901.79 loss:0.026 grdn:0.560 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 19:42:50 ot_train.py:423 step:63K smpl:32M ep:36K epch:904.65 loss:0.026 grdn:0.574 lr:5.0e-05 updt_s:0.326 data_s:0.025
INFO 2026-08-10 19:43:59 ot_train.py:423 step:63K smpl:32M ep:36K epch:907.51 loss:0.026 grdn:0.572 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 19:45:08 ot_train.py:423 step:64K smpl:33M ep:36K epch:910.37 loss:0.026 grdn:inf lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 19:46:19 ot_train.py:423 step:64K smpl:33M ep:37K epch:913.24 loss:0.026 grdn:0.591 lr:5.0e-05 updt_s:0.324 data_s:0.026
INFO 2026-08-10 19:47:28 ot_train.py:423 step:64K smpl:33M ep:37K epch:916.10 loss:0.025 grdn:0.569 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 19:47:28 ot_train.py:443 Checkpoint policy after step 64000
INFO 2026-08-10 19:48:38 ot_train.py:423 step:64K smpl:33M ep:37K epch:918.96 loss:0.026 grdn:0.597 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 19:49:48 ot_train.py:423 step:64K smpl:33M ep:37K epch:921.83 loss:0.026 grdn:0.569 lr:5.0e-05 updt_s:0.325 data_s:0.026
INFO 2026-08-10 19:50:58 ot_train.py:423 step:65K smpl:33M ep:37K epch:924.69 loss:0.026 grdn:0.586 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 19:52:07 ot_train.py:423 step:65K smpl:33M ep:37K epch:927.55 loss:0.025 grdn:0.584 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 19:53:17 ot_train.py:423 step:65K smpl:33M ep:37K epch:930.41 loss:0.025 grdn:0.549 lr:5.0e-05 updt_s:0.325 data_s:0.026
INFO 2026-08-10 19:54:26 ot_train.py:423 step:65K smpl:33M ep:37K epch:933.28 loss:0.026 grdn:0.569 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 19:55:35 ot_train.py:423 step:65K smpl:33M ep:37K epch:936.14 loss:0.025 grdn:0.549 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-10 19:56:46 ot_train.py:423 step:66K smpl:34M ep:38K epch:939.00 loss:0.025 grdn:inf lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-10 19:57:55 ot_train.py:423 step:66K smpl:34M ep:38K epch:941.87 loss:0.025 grdn:0.549 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 19:59:06 ot_train.py:423 step:66K smpl:34M ep:38K epch:944.73 loss:0.025 grdn:0.604 lr:5.0e-05 updt_s:0.329 data_s:0.025
INFO 2026-08-10 19:59:06 ot_train.py:443 Checkpoint policy after step 66000
INFO 2026-08-10 20:00:16 ot_train.py:423 step:66K smpl:34M ep:38K epch:947.59 loss:0.026 grdn:0.572 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 20:01:25 ot_train.py:423 step:66K smpl:34M ep:38K epch:950.45 loss:0.025 grdn:0.561 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 20:02:36 ot_train.py:423 step:67K smpl:34M ep:38K epch:953.32 loss:0.025 grdn:0.537 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-10 20:03:45 ot_train.py:423 step:67K smpl:34M ep:38K epch:956.18 loss:0.026 grdn:0.574 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 20:04:55 ot_train.py:423 step:67K smpl:34M ep:38K epch:959.04 loss:0.025 grdn:0.556 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-10 20:06:05 ot_train.py:423 step:67K smpl:34M ep:38K epch:961.91 loss:0.025 grdn:0.569 lr:5.0e-05 updt_s:0.327 data_s:0.025
INFO 2026-08-10 20:07:14 ot_train.py:423 step:67K smpl:35M ep:39K epch:964.77 loss:0.025 grdn:0.542 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 20:08:23 ot_train.py:423 step:68K smpl:35M ep:39K epch:967.63 loss:0.026 grdn:inf lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 20:09:34 ot_train.py:423 step:68K smpl:35M ep:39K epch:970.49 loss:0.025 grdn:0.548 lr:5.0e-05 updt_s:0.325 data_s:0.026
INFO 2026-08-10 20:10:43 ot_train.py:423 step:68K smpl:35M ep:39K epch:973.36 loss:0.025 grdn:0.549 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 20:10:43 ot_train.py:443 Checkpoint policy after step 68000
INFO 2026-08-10 20:11:54 ot_train.py:423 step:68K smpl:35M ep:39K epch:976.22 loss:0.025 grdn:0.548 lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-10 20:13:03 ot_train.py:423 step:68K smpl:35M ep:39K epch:979.08 loss:0.025 grdn:0.600 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 20:14:12 ot_train.py:423 step:69K smpl:35M ep:39K epch:981.95 loss:0.025 grdn:0.550 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 20:15:23 ot_train.py:423 step:69K smpl:35M ep:39K epch:984.81 loss:0.025 grdn:0.575 lr:5.0e-05 updt_s:0.327 data_s:0.025
INFO 2026-08-10 20:16:32 ot_train.py:423 step:69K smpl:35M ep:40K epch:987.67 loss:0.024 grdn:0.564 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 20:17:41 ot_train.py:423 step:69K smpl:35M ep:40K epch:990.53 loss:0.025 grdn:0.560 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 20:18:52 ot_train.py:423 step:69K smpl:36M ep:40K epch:993.40 loss:0.025 grdn:0.535 lr:5.0e-05 updt_s:0.327 data_s:0.025
INFO 2026-08-10 20:20:01 ot_train.py:423 step:70K smpl:36M ep:40K epch:996.26 loss:0.025 grdn:inf lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 20:21:10 ot_train.py:423 step:70K smpl:36M ep:40K epch:999.12 loss:0.025 grdn:0.558 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 20:22:21 ot_train.py:423 step:70K smpl:36M ep:40K epch:1001.98 loss:0.024 grdn:0.561 lr:5.0e-05 updt_s:0.329 data_s:0.025
INFO 2026-08-10 20:22:21 ot_train.py:443 Checkpoint policy after step 70000
INFO 2026-08-10 20:23:30 ot_train.py:423 step:70K smpl:36M ep:40K epch:1004.85 loss:0.025 grdn:0.566 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-10 20:24:40 ot_train.py:423 step:70K smpl:36M ep:40K epch:1007.71 loss:0.024 grdn:0.555 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 20:25:51 ot_train.py:423 step:71K smpl:36M ep:40K epch:1010.57 loss:0.025 grdn:0.590 lr:5.0e-05 updt_s:0.329 data_s:0.026
INFO 2026-08-10 20:27:00 ot_train.py:423 step:71K smpl:36M ep:41K epch:1013.44 loss:0.025 grdn:0.541 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 20:28:11 ot_train.py:423 step:71K smpl:36M ep:41K epch:1016.30 loss:0.025 grdn:0.565 lr:5.0e-05 updt_s:0.326 data_s:0.027
INFO 2026-08-10 20:29:20 ot_train.py:423 step:71K smpl:36M ep:41K epch:1019.16 loss:0.025 grdn:0.564 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-10 20:30:30 ot_train.py:423 step:71K smpl:37M ep:41K epch:1022.02 loss:0.024 grdn:0.524 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 20:31:41 ot_train.py:423 step:72K smpl:37M ep:41K epch:1024.89 loss:0.024 grdn:0.533 lr:5.0e-05 updt_s:0.329 data_s:0.026
INFO 2026-08-10 20:32:50 ot_train.py:423 step:72K smpl:37M ep:41K epch:1027.75 loss:0.025 grdn:inf lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 20:33:59 ot_train.py:423 step:72K smpl:37M ep:41K epch:1030.61 loss:0.025 grdn:0.583 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 20:33:59 ot_train.py:443 Checkpoint policy after step 72000
INFO 2026-08-10 20:35:11 ot_train.py:423 step:72K smpl:37M ep:41K epch:1033.48 loss:0.025 grdn:0.586 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-10 20:36:20 ot_train.py:423 step:72K smpl:37M ep:41K epch:1036.34 loss:0.024 grdn:0.527 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-10 20:37:29 ot_train.py:423 step:73K smpl:37M ep:42K epch:1039.20 loss:0.024 grdn:0.546 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 20:38:40 ot_train.py:423 step:73K smpl:37M ep:42K epch:1042.06 loss:0.024 grdn:0.533 lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-10 20:39:49 ot_train.py:423 step:73K smpl:37M ep:42K epch:1044.93 loss:0.024 grdn:0.567 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 20:40:58 ot_train.py:423 step:73K smpl:37M ep:42K epch:1047.79 loss:0.024 grdn:0.534 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 20:42:09 ot_train.py:423 step:73K smpl:38M ep:42K epch:1050.65 loss:0.024 grdn:0.548 lr:5.0e-05 updt_s:0.329 data_s:0.026
INFO 2026-08-10 20:43:17 ot_train.py:423 step:74K smpl:38M ep:42K epch:1053.52 loss:0.024 grdn:0.536 lr:5.0e-05 updt_s:0.321 data_s:0.019
INFO 2026-08-10 20:44:28 ot_train.py:423 step:74K smpl:38M ep:42K epch:1056.38 loss:0.024 grdn:inf lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-10 20:45:37 ot_train.py:423 step:74K smpl:38M ep:42K epch:1059.24 loss:0.024 grdn:0.552 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 20:45:37 ot_train.py:443 Checkpoint policy after step 74000
INFO 2026-08-10 20:46:47 ot_train.py:423 step:74K smpl:38M ep:42K epch:1062.10 loss:0.024 grdn:0.579 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 20:47:58 ot_train.py:423 step:74K smpl:38M ep:43K epch:1064.97 loss:0.024 grdn:0.518 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-10 20:49:07 ot_train.py:423 step:75K smpl:38M ep:43K epch:1067.83 loss:0.024 grdn:0.546 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 20:50:15 ot_train.py:423 step:75K smpl:38M ep:43K epch:1070.69 loss:0.024 grdn:0.541 lr:5.0e-05 updt_s:0.322 data_s:0.019
INFO 2026-08-10 20:51:26 ot_train.py:423 step:75K smpl:38M ep:43K epch:1073.56 loss:0.024 grdn:0.553 lr:5.0e-05 updt_s:0.329 data_s:0.025
INFO 2026-08-10 20:52:35 ot_train.py:423 step:75K smpl:39M ep:43K epch:1076.42 loss:0.024 grdn:0.580 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 20:53:44 ot_train.py:423 step:75K smpl:39M ep:43K epch:1079.28 loss:0.024 grdn:0.586 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 20:54:55 ot_train.py:423 step:76K smpl:39M ep:43K epch:1082.14 loss:0.024 grdn:0.561 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-10 20:56:04 ot_train.py:423 step:76K smpl:39M ep:43K epch:1085.01 loss:0.024 grdn:0.560 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 20:57:13 ot_train.py:423 step:76K smpl:39M ep:44K epch:1087.87 loss:0.024 grdn:inf lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 20:57:13 ot_train.py:443 Checkpoint policy after step 76000
INFO 2026-08-10 20:58:24 ot_train.py:423 step:76K smpl:39M ep:44K epch:1090.73 loss:0.024 grdn:0.533 lr:5.0e-05 updt_s:0.325 data_s:0.026
INFO 2026-08-10 20:59:33 ot_train.py:423 step:76K smpl:39M ep:44K epch:1093.60 loss:0.024 grdn:0.557 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 21:00:44 ot_train.py:423 step:77K smpl:39M ep:44K epch:1096.46 loss:0.024 grdn:0.521 lr:5.0e-05 updt_s:0.325 data_s:0.026
INFO 2026-08-10 21:01:53 ot_train.py:423 step:77K smpl:39M ep:44K epch:1099.32 loss:0.023 grdn:0.527 lr:5.0e-05 updt_s:0.329 data_s:0.019
INFO 2026-08-10 21:03:02 ot_train.py:423 step:77K smpl:39M ep:44K epch:1102.18 loss:0.024 grdn:0.548 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-10 21:04:13 ot_train.py:423 step:77K smpl:40M ep:44K epch:1105.05 loss:0.024 grdn:0.563 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-10 21:05:22 ot_train.py:423 step:77K smpl:40M ep:44K epch:1107.91 loss:0.024 grdn:0.516 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 21:06:31 ot_train.py:423 step:78K smpl:40M ep:44K epch:1110.77 loss:0.024 grdn:0.570 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 21:07:41 ot_train.py:423 step:78K smpl:40M ep:45K epch:1113.63 loss:0.023 grdn:0.535 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-10 21:08:50 ot_train.py:423 step:78K smpl:40M ep:45K epch:1116.50 loss:0.023 grdn:inf lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 21:08:50 ot_train.py:443 Checkpoint policy after step 78000
INFO 2026-08-10 21:10:00 ot_train.py:423 step:78K smpl:40M ep:45K epch:1119.36 loss:0.024 grdn:0.544 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 21:11:11 ot_train.py:423 step:78K smpl:40M ep:45K epch:1122.22 loss:0.023 grdn:0.564 lr:5.0e-05 updt_s:0.329 data_s:0.025
INFO 2026-08-10 21:12:20 ot_train.py:423 step:79K smpl:40M ep:45K epch:1125.09 loss:0.023 grdn:0.532 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-10 21:13:29 ot_train.py:423 step:79K smpl:40M ep:45K epch:1127.95 loss:0.023 grdn:0.531 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 21:14:39 ot_train.py:423 step:79K smpl:40M ep:45K epch:1130.81 loss:0.023 grdn:0.574 lr:5.0e-05 updt_s:0.326 data_s:0.024
INFO 2026-08-10 21:15:49 ot_train.py:423 step:79K smpl:41M ep:45K epch:1133.67 loss:0.023 grdn:0.544 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-10 21:16:59 ot_train.py:423 step:79K smpl:41M ep:45K epch:1136.54 loss:0.024 grdn:0.535 lr:5.0e-05 updt_s:0.325 data_s:0.025
INFO 2026-08-10 21:18:08 ot_train.py:423 step:80K smpl:41M ep:46K epch:1139.40 loss:0.023 grdn:0.551 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-10 21:19:17 ot_train.py:423 step:80K smpl:41M ep:46K epch:1142.26 loss:0.024 grdn:0.525 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-10 21:20:28 ot_train.py:423 step:80K smpl:41M ep:46K epch:1145.13 loss:0.023 grdn:inf lr:5.0e-05 updt_s:0.327 data_s:0.027
INFO 2026-08-10 21:20:28 ot_train.py:443 Checkpoint policy after step 80000
INFO 2026-08-10 21:21:37 ot_train.py:423 step:80K smpl:41M ep:46K epch:1147.99 loss:0.023 grdn:0.540 lr:5.0e-05 updt_s:0.322 data_s:0.019
INFO 2026-08-10 21:22:46 ot_train.py:423 step:80K smpl:41M ep:46K epch:1150.85 loss:0.023 grdn:0.555 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 21:23:57 ot_train.py:423 step:81K smpl:41M ep:46K epch:1153.71 loss:0.023 grdn:0.569 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-10 21:25:06 ot_train.py:423 step:81K smpl:41M ep:46K epch:1156.58 loss:0.024 grdn:0.550 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 21:26:15 ot_train.py:423 step:81K smpl:41M ep:46K epch:1159.44 loss:0.023 grdn:0.538 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 21:27:26 ot_train.py:423 step:81K smpl:42M ep:46K epch:1162.30 loss:0.023 grdn:0.520 lr:5.0e-05 updt_s:0.325 data_s:0.026
INFO 2026-08-10 21:28:35 ot_train.py:423 step:81K smpl:42M ep:47K epch:1165.17 loss:0.023 grdn:0.556 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 21:29:43 ot_train.py:423 step:82K smpl:42M ep:47K epch:1168.03 loss:0.023 grdn:0.566 lr:5.0e-05 updt_s:0.322 data_s:0.019
INFO 2026-08-10 21:30:55 ot_train.py:423 step:82K smpl:42M ep:47K epch:1170.89 loss:0.023 grdn:0.559 lr:5.0e-05 updt_s:0.330 data_s:0.027
INFO 2026-08-10 21:32:04 ot_train.py:423 step:82K smpl:42M ep:47K epch:1173.75 loss:0.023 grdn:inf lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 21:32:04 ot_train.py:443 Checkpoint policy after step 82000
INFO 2026-08-10 21:33:15 ot_train.py:423 step:82K smpl:42M ep:47K epch:1176.62 loss:0.024 grdn:0.540 lr:5.0e-05 updt_s:0.327 data_s:0.025
INFO 2026-08-10 21:34:25 ot_train.py:423 step:82K smpl:42M ep:47K epch:1179.48 loss:0.023 grdn:0.530 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 21:35:34 ot_train.py:423 step:83K smpl:42M ep:47K epch:1182.34 loss:0.023 grdn:0.526 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 21:36:45 ot_train.py:423 step:83K smpl:42M ep:47K epch:1185.21 loss:0.023 grdn:0.531 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-10 21:37:54 ot_train.py:423 step:83K smpl:42M ep:48K epch:1188.07 loss:0.023 grdn:0.524 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 21:39:03 ot_train.py:423 step:83K smpl:43M ep:48K epch:1190.93 loss:0.023 grdn:0.572 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-10 21:40:13 ot_train.py:423 step:83K smpl:43M ep:48K epch:1193.79 loss:0.023 grdn:0.563 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-10 21:41:23 ot_train.py:423 step:84K smpl:43M ep:48K epch:1196.66 loss:0.023 grdn:0.559 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-10 21:42:32 ot_train.py:423 step:84K smpl:43M ep:48K epch:1199.52 loss:0.023 grdn:0.529 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-10 21:43:42 ot_train.py:423 step:84K smpl:43M ep:48K epch:1202.38 loss:0.023 grdn:0.512 lr:5.0e-05 updt_s:0.327 data_s:0.025
INFO 2026-08-10 21:43:42 ot_train.py:443 Checkpoint policy after step 84000
INFO 2026-08-10 21:44:52 ot_train.py:423 step:84K smpl:43M ep:48K epch:1205.24 loss:0.022 grdn:inf lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 21:46:01 ot_train.py:423 step:84K smpl:43M ep:48K epch:1208.11 loss:0.023 grdn:0.518 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 21:47:12 ot_train.py:423 step:85K smpl:43M ep:48K epch:1210.97 loss:0.023 grdn:0.576 lr:5.0e-05 updt_s:0.326 data_s:0.025
INFO 2026-08-10 21:48:21 ot_train.py:423 step:85K smpl:43M ep:49K epch:1213.83 loss:0.023 grdn:0.543 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 21:49:32 ot_train.py:423 step:85K smpl:44M ep:49K epch:1216.70 loss:0.022 grdn:0.526 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-10 21:50:41 ot_train.py:423 step:85K smpl:44M ep:49K epch:1219.56 loss:0.022 grdn:0.537 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-10 21:51:50 ot_train.py:423 step:85K smpl:44M ep:49K epch:1222.42 loss:0.023 grdn:0.503 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-10 21:53:01 ot_train.py:423 step:86K smpl:44M ep:49K epch:1225.28 loss:0.022 grdn:0.550 lr:5.0e-05 updt_s:0.329 data_s:0.026
INFO 2026-08-10 21:54:10 ot_train.py:423 step:86K smpl:44M ep:49K epch:1228.15 loss:0.023 grdn:0.539 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 21:55:19 ot_train.py:423 step:86K smpl:44M ep:49K epch:1231.01 loss:0.023 grdn:0.541 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 21:55:19 ot_train.py:443 Checkpoint policy after step 86000
INFO 2026-08-10 21:56:30 ot_train.py:423 step:86K smpl:44M ep:49K epch:1233.87 loss:0.022 grdn:inf lr:5.0e-05 updt_s:0.325 data_s:0.026
INFO 2026-08-10 21:57:39 ot_train.py:423 step:86K smpl:44M ep:49K epch:1236.74 loss:0.022 grdn:0.550 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 21:58:49 ot_train.py:423 step:87K smpl:44M ep:50K epch:1239.60 loss:0.022 grdn:0.548 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 22:00:00 ot_train.py:423 step:87K smpl:44M ep:50K epch:1242.46 loss:0.022 grdn:0.526 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-10 22:01:09 ot_train.py:423 step:87K smpl:45M ep:50K epch:1245.32 loss:0.023 grdn:0.576 lr:5.0e-05 updt_s:0.330 data_s:0.019
INFO 2026-08-10 22:02:18 ot_train.py:423 step:87K smpl:45M ep:50K epch:1248.19 loss:0.022 grdn:0.518 lr:5.0e-05 updt_s:0.322 data_s:0.019
INFO 2026-08-10 22:03:30 ot_train.py:423 step:87K smpl:45M ep:50K epch:1251.05 loss:0.023 grdn:0.530 lr:5.0e-05 updt_s:0.332 data_s:0.026
INFO 2026-08-10 22:04:39 ot_train.py:423 step:88K smpl:45M ep:50K epch:1253.91 loss:0.023 grdn:0.528 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 22:05:49 ot_train.py:423 step:88K smpl:45M ep:50K epch:1256.78 loss:0.022 grdn:0.524 lr:5.0e-05 updt_s:0.327 data_s:0.025
INFO 2026-08-10 22:06:59 ot_train.py:423 step:88K smpl:45M ep:50K epch:1259.64 loss:0.023 grdn:0.539 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-10 22:06:59 ot_train.py:443 Checkpoint policy after step 88000
INFO 2026-08-10 22:08:09 ot_train.py:423 step:88K smpl:45M ep:51K epch:1262.50 loss:0.022 grdn:inf lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 22:09:19 ot_train.py:423 step:88K smpl:45M ep:51K epch:1265.36 loss:0.022 grdn:0.518 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-10 22:10:29 ot_train.py:423 step:89K smpl:45M ep:51K epch:1268.23 loss:0.023 grdn:0.506 lr:5.0e-05 updt_s:0.330 data_s:0.019
INFO 2026-08-10 22:11:38 ot_train.py:423 step:89K smpl:45M ep:51K epch:1271.09 loss:0.022 grdn:0.551 lr:5.0e-05 updt_s:0.323 data_s:0.019
INFO 2026-08-10 22:12:49 ot_train.py:423 step:89K smpl:46M ep:51K epch:1273.95 loss:0.022 grdn:0.573 lr:5.0e-05 updt_s:0.329 data_s:0.026
INFO 2026-08-10 22:13:58 ot_train.py:423 step:89K smpl:46M ep:51K epch:1276.82 loss:0.022 grdn:0.546 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 22:15:07 ot_train.py:423 step:89K smpl:46M ep:51K epch:1279.68 loss:0.023 grdn:0.532 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 22:16:18 ot_train.py:423 step:90K smpl:46M ep:51K epch:1282.54 loss:0.022 grdn:0.512 lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-10 22:17:27 ot_train.py:423 step:90K smpl:46M ep:51K epch:1285.40 loss:0.022 grdn:0.513 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-10 22:18:38 ot_train.py:423 step:90K smpl:46M ep:52K epch:1288.27 loss:0.022 grdn:0.535 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-10 22:18:38 ot_train.py:443 Checkpoint policy after step 90000
INFO 2026-08-10 22:19:48 ot_train.py:423 step:90K smpl:46M ep:52K epch:1291.13 loss:0.022 grdn:0.544 lr:5.0e-05 updt_s:0.329 data_s:0.019
INFO 2026-08-10 22:20:57 ot_train.py:423 step:90K smpl:46M ep:52K epch:1293.99 loss:0.022 grdn:inf lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-10 22:22:08 ot_train.py:423 step:91K smpl:46M ep:52K epch:1296.85 loss:0.022 grdn:0.520 lr:5.0e-05 updt_s:0.330 data_s:0.026
INFO 2026-08-10 22:23:17 ot_train.py:423 step:91K smpl:46M ep:52K epch:1299.72 loss:0.022 grdn:0.543 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 22:24:27 ot_train.py:423 step:91K smpl:47M ep:52K epch:1302.58 loss:0.022 grdn:0.543 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 22:25:37 ot_train.py:423 step:91K smpl:47M ep:52K epch:1305.44 loss:0.022 grdn:0.567 lr:5.0e-05 updt_s:0.325 data_s:0.026
INFO 2026-08-10 22:26:47 ot_train.py:423 step:91K smpl:47M ep:52K epch:1308.31 loss:0.022 grdn:0.529 lr:5.0e-05 updt_s:0.330 data_s:0.019
INFO 2026-08-10 22:27:56 ot_train.py:423 step:92K smpl:47M ep:52K epch:1311.17 loss:0.022 grdn:0.525 lr:5.0e-05 updt_s:0.325 data_s:0.020
INFO 2026-08-10 22:29:07 ot_train.py:423 step:92K smpl:47M ep:53K epch:1314.03 loss:0.022 grdn:0.518 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-10 22:30:16 ot_train.py:423 step:92K smpl:47M ep:53K epch:1316.89 loss:0.022 grdn:0.538 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 22:30:16 ot_train.py:443 Checkpoint policy after step 92000
INFO 2026-08-10 22:31:26 ot_train.py:423 step:92K smpl:47M ep:53K epch:1319.76 loss:0.022 grdn:0.548 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 22:32:37 ot_train.py:423 step:92K smpl:47M ep:53K epch:1322.62 loss:0.022 grdn:inf lr:5.0e-05 updt_s:0.329 data_s:0.026
INFO 2026-08-10 22:33:46 ot_train.py:423 step:93K smpl:47M ep:53K epch:1325.48 loss:0.022 grdn:0.527 lr:5.0e-05 updt_s:0.323 data_s:0.019
INFO 2026-08-10 22:34:57 ot_train.py:423 step:93K smpl:48M ep:53K epch:1328.35 loss:0.022 grdn:0.525 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-10 22:36:06 ot_train.py:423 step:93K smpl:48M ep:53K epch:1331.21 loss:0.022 grdn:0.520 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 22:37:15 ot_train.py:423 step:93K smpl:48M ep:53K epch:1334.07 loss:0.022 grdn:0.558 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 22:38:26 ot_train.py:423 step:93K smpl:48M ep:53K epch:1336.93 loss:0.022 grdn:0.536 lr:5.0e-05 updt_s:0.329 data_s:0.025
INFO 2026-08-10 22:39:35 ot_train.py:423 step:94K smpl:48M ep:54K epch:1339.80 loss:0.022 grdn:0.557 lr:5.0e-05 updt_s:0.323 data_s:0.019
INFO 2026-08-10 22:40:44 ot_train.py:423 step:94K smpl:48M ep:54K epch:1342.66 loss:0.022 grdn:0.519 lr:5.0e-05 updt_s:0.326 data_s:0.020
INFO 2026-08-10 22:41:55 ot_train.py:423 step:94K smpl:48M ep:54K epch:1345.52 loss:0.022 grdn:0.547 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-10 22:41:55 ot_train.py:443 Checkpoint policy after step 94000
INFO 2026-08-10 22:43:05 ot_train.py:423 step:94K smpl:48M ep:54K epch:1348.39 loss:0.022 grdn:0.499 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 22:44:14 ot_train.py:423 step:94K smpl:48M ep:54K epch:1351.25 loss:0.022 grdn:0.491 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 22:45:24 ot_train.py:423 step:95K smpl:48M ep:54K epch:1354.11 loss:0.022 grdn:inf lr:5.0e-05 updt_s:0.329 data_s:0.024
INFO 2026-08-10 22:46:33 ot_train.py:423 step:95K smpl:49M ep:54K epch:1356.97 loss:0.022 grdn:0.530 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 22:47:42 ot_train.py:423 step:95K smpl:49M ep:54K epch:1359.84 loss:0.022 grdn:0.529 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 22:48:53 ot_train.py:423 step:95K smpl:49M ep:55K epch:1362.70 loss:0.022 grdn:0.522 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-10 22:50:02 ot_train.py:423 step:95K smpl:49M ep:55K epch:1365.56 loss:0.022 grdn:0.526 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 22:51:13 ot_train.py:423 step:96K smpl:49M ep:55K epch:1368.43 loss:0.021 grdn:0.540 lr:5.0e-05 updt_s:0.330 data_s:0.025
INFO 2026-08-10 22:52:23 ot_train.py:423 step:96K smpl:49M ep:55K epch:1371.29 loss:0.021 grdn:0.513 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-10 22:53:32 ot_train.py:423 step:96K smpl:49M ep:55K epch:1374.15 loss:0.022 grdn:0.500 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 22:53:32 ot_train.py:443 Checkpoint policy after step 96000
INFO 2026-08-10 22:54:43 ot_train.py:423 step:96K smpl:49M ep:55K epch:1377.01 loss:0.021 grdn:0.539 lr:5.0e-05 updt_s:0.325 data_s:0.026
INFO 2026-08-10 22:55:52 ot_train.py:423 step:96K smpl:49M ep:55K epch:1379.88 loss:0.021 grdn:0.528 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 22:57:01 ot_train.py:423 step:97K smpl:49M ep:55K epch:1382.74 loss:0.022 grdn:inf lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 22:58:12 ot_train.py:423 step:97K smpl:50M ep:55K epch:1385.60 loss:0.022 grdn:0.543 lr:5.0e-05 updt_s:0.323 data_s:0.029
INFO 2026-08-10 22:59:21 ot_train.py:423 step:97K smpl:50M ep:56K epch:1388.46 loss:0.021 grdn:0.542 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 23:00:30 ot_train.py:423 step:97K smpl:50M ep:56K epch:1391.33 loss:0.021 grdn:0.529 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 23:01:41 ot_train.py:423 step:97K smpl:50M ep:56K epch:1394.19 loss:0.022 grdn:0.512 lr:5.0e-05 updt_s:0.328 data_s:0.027
INFO 2026-08-10 23:02:51 ot_train.py:423 step:98K smpl:50M ep:56K epch:1397.05 loss:0.021 grdn:0.538 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 23:03:59 ot_train.py:423 step:98K smpl:50M ep:56K epch:1399.92 loss:0.021 grdn:0.519 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-10 23:05:10 ot_train.py:423 step:98K smpl:50M ep:56K epch:1402.78 loss:0.021 grdn:0.526 lr:5.0e-05 updt_s:0.328 data_s:0.025
INFO 2026-08-10 23:05:10 ot_train.py:443 Checkpoint policy after step 98000
INFO 2026-08-10 23:06:20 ot_train.py:423 step:98K smpl:50M ep:56K epch:1405.64 loss:0.022 grdn:0.572 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 23:07:30 ot_train.py:423 step:98K smpl:50M ep:56K epch:1408.50 loss:0.021 grdn:0.506 lr:5.0e-05 updt_s:0.326 data_s:0.027
INFO 2026-08-10 23:08:40 ot_train.py:423 step:99K smpl:50M ep:56K epch:1411.37 loss:0.021 grdn:0.523 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 23:09:49 ot_train.py:423 step:99K smpl:51M ep:57K epch:1414.23 loss:0.022 grdn:inf lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 23:10:59 ot_train.py:423 step:99K smpl:51M ep:57K epch:1417.09 loss:0.022 grdn:0.536 lr:5.0e-05 updt_s:0.326 data_s:0.025
INFO 2026-08-10 23:12:08 ot_train.py:423 step:99K smpl:51M ep:57K epch:1419.96 loss:0.021 grdn:0.548 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-10 23:13:18 ot_train.py:423 step:99K smpl:51M ep:57K epch:1422.82 loss:0.021 grdn:0.538 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 23:14:28 ot_train.py:423 step:100K smpl:51M ep:57K epch:1425.68 loss:0.021 grdn:0.524 lr:5.0e-05 updt_s:0.325 data_s:0.027
INFO 2026-08-10 23:15:38 ot_train.py:423 step:100K smpl:51M ep:57K epch:1428.54 loss:0.022 grdn:0.514 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 23:16:47 ot_train.py:423 step:100K smpl:51M ep:57K epch:1431.41 loss:0.021 grdn:0.507 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-10 23:16:47 ot_train.py:443 Checkpoint policy after step 100000
INFO 2026-08-10 23:17:58 ot_train.py:423 step:100K smpl:51M ep:57K epch:1434.27 loss:0.021 grdn:0.529 lr:5.0e-05 updt_s:0.329 data_s:0.025
INFO 2026-08-10 23:19:08 ot_train.py:423 step:100K smpl:51M ep:57K epch:1437.13 loss:0.021 grdn:0.526 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 23:20:17 ot_train.py:423 step:101K smpl:52M ep:58K epch:1440.00 loss:0.021 grdn:0.497 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 23:21:27 ot_train.py:423 step:101K smpl:52M ep:58K epch:1442.86 loss:0.021 grdn:0.531 lr:5.0e-05 updt_s:0.325 data_s:0.026
INFO 2026-08-10 23:22:37 ot_train.py:423 step:101K smpl:52M ep:58K epch:1445.72 loss:0.021 grdn:inf lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-10 23:23:47 ot_train.py:423 step:101K smpl:52M ep:58K epch:1448.58 loss:0.021 grdn:0.519 lr:5.0e-05 updt_s:0.327 data_s:0.024
INFO 2026-08-10 23:24:57 ot_train.py:423 step:101K smpl:52M ep:58K epch:1451.45 loss:0.021 grdn:0.513 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-10 23:26:06 ot_train.py:423 step:102K smpl:52M ep:58K epch:1454.31 loss:0.021 grdn:0.537 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 23:27:18 ot_train.py:423 step:102K smpl:52M ep:58K epch:1457.17 loss:0.021 grdn:0.530 lr:5.0e-05 updt_s:0.330 data_s:0.025
INFO 2026-08-10 23:28:26 ot_train.py:423 step:102K smpl:52M ep:58K epch:1460.04 loss:0.021 grdn:0.564 lr:5.0e-05 updt_s:0.323 data_s:0.019
INFO 2026-08-10 23:28:26 ot_train.py:443 Checkpoint policy after step 102000
INFO 2026-08-10 23:29:36 ot_train.py:423 step:102K smpl:52M ep:59K epch:1462.90 loss:0.021 grdn:0.508 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 23:30:47 ot_train.py:423 step:102K smpl:52M ep:59K epch:1465.76 loss:0.021 grdn:0.548 lr:5.0e-05 updt_s:0.329 data_s:0.026
INFO 2026-08-10 23:31:56 ot_train.py:423 step:103K smpl:53M ep:59K epch:1468.62 loss:0.021 grdn:0.555 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-10 23:33:05 ot_train.py:423 step:103K smpl:53M ep:59K epch:1471.49 loss:0.021 grdn:0.512 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 23:34:16 ot_train.py:423 step:103K smpl:53M ep:59K epch:1474.35 loss:0.021 grdn:inf lr:5.0e-05 updt_s:0.327 data_s:0.025
INFO 2026-08-10 23:35:25 ot_train.py:423 step:103K smpl:53M ep:59K epch:1477.21 loss:0.021 grdn:0.559 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-10 23:36:35 ot_train.py:423 step:103K smpl:53M ep:59K epch:1480.07 loss:0.021 grdn:0.526 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 23:37:46 ot_train.py:423 step:104K smpl:53M ep:59K epch:1482.94 loss:0.021 grdn:0.506 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-10 23:38:54 ot_train.py:423 step:104K smpl:53M ep:59K epch:1485.80 loss:0.021 grdn:0.509 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-10 23:40:05 ot_train.py:423 step:104K smpl:53M ep:60K epch:1488.66 loss:0.021 grdn:0.529 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-10 23:40:05 ot_train.py:443 Checkpoint policy after step 104000
INFO 2026-08-10 23:41:15 ot_train.py:423 step:104K smpl:53M ep:60K epch:1491.53 loss:0.021 grdn:0.526 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 23:42:24 ot_train.py:423 step:104K smpl:53M ep:60K epch:1494.39 loss:0.021 grdn:0.516 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 23:43:35 ot_train.py:423 step:105K smpl:54M ep:60K epch:1497.25 loss:0.021 grdn:0.506 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-10 23:44:44 ot_train.py:423 step:105K smpl:54M ep:60K epch:1500.11 loss:0.021 grdn:0.520 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 23:45:53 ot_train.py:423 step:105K smpl:54M ep:60K epch:1502.98 loss:0.021 grdn:0.483 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 23:47:04 ot_train.py:423 step:105K smpl:54M ep:60K epch:1505.84 loss:0.021 grdn:inf lr:5.0e-05 updt_s:0.330 data_s:0.025
INFO 2026-08-10 23:48:13 ot_train.py:423 step:105K smpl:54M ep:60K epch:1508.70 loss:0.021 grdn:0.523 lr:5.0e-05 updt_s:0.323 data_s:0.019
INFO 2026-08-10 23:49:22 ot_train.py:423 step:106K smpl:54M ep:60K epch:1511.57 loss:0.021 grdn:0.522 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 23:50:33 ot_train.py:423 step:106K smpl:54M ep:61K epch:1514.43 loss:0.021 grdn:0.520 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-10 23:51:42 ot_train.py:423 step:106K smpl:54M ep:61K epch:1517.29 loss:0.020 grdn:0.490 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-10 23:51:42 ot_train.py:443 Checkpoint policy after step 106000
INFO 2026-08-10 23:52:52 ot_train.py:423 step:106K smpl:54M ep:61K epch:1520.15 loss:0.020 grdn:0.532 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-10 23:54:03 ot_train.py:423 step:106K smpl:54M ep:61K epch:1523.02 loss:0.020 grdn:0.560 lr:5.0e-05 updt_s:0.331 data_s:0.025
INFO 2026-08-10 23:55:12 ot_train.py:423 step:107K smpl:55M ep:61K epch:1525.88 loss:0.021 grdn:0.546 lr:5.0e-05 updt_s:0.323 data_s:0.019
INFO 2026-08-10 23:56:23 ot_train.py:423 step:107K smpl:55M ep:61K epch:1528.74 loss:0.021 grdn:0.508 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-10 23:57:33 ot_train.py:423 step:107K smpl:55M ep:61K epch:1531.61 loss:0.020 grdn:0.523 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-10 23:58:42 ot_train.py:423 step:107K smpl:55M ep:61K epch:1534.47 loss:0.020 grdn:0.518 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-10 23:59:52 ot_train.py:423 step:107K smpl:55M ep:61K epch:1537.33 loss:0.021 grdn:inf lr:5.0e-05 updt_s:0.327 data_s:0.025
INFO 2026-08-11 00:01:02 ot_train.py:423 step:108K smpl:55M ep:62K epch:1540.19 loss:0.021 grdn:0.525 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 00:02:11 ot_train.py:423 step:108K smpl:55M ep:62K epch:1543.06 loss:0.021 grdn:0.570 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 00:03:22 ot_train.py:423 step:108K smpl:55M ep:62K epch:1545.92 loss:0.021 grdn:0.502 lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-11 00:03:22 ot_train.py:443 Checkpoint policy after step 108000
INFO 2026-08-11 00:04:32 ot_train.py:423 step:108K smpl:55M ep:62K epch:1548.78 loss:0.021 grdn:0.532 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 00:05:41 ot_train.py:423 step:108K smpl:56M ep:62K epch:1551.65 loss:0.020 grdn:0.517 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 00:06:52 ot_train.py:423 step:109K smpl:56M ep:62K epch:1554.51 loss:0.020 grdn:0.567 lr:5.0e-05 updt_s:0.330 data_s:0.026
INFO 2026-08-11 00:08:01 ot_train.py:423 step:109K smpl:56M ep:62K epch:1557.37 loss:0.020 grdn:0.507 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 00:09:10 ot_train.py:423 step:109K smpl:56M ep:62K epch:1560.23 loss:0.021 grdn:0.499 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 00:10:21 ot_train.py:423 step:109K smpl:56M ep:63K epch:1563.10 loss:0.021 grdn:0.536 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-11 00:11:30 ot_train.py:423 step:109K smpl:56M ep:63K epch:1565.96 loss:0.020 grdn:inf lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 00:12:41 ot_train.py:423 step:110K smpl:56M ep:63K epch:1568.82 loss:0.021 grdn:0.515 lr:5.0e-05 updt_s:0.325 data_s:0.026
INFO 2026-08-11 00:13:50 ot_train.py:423 step:110K smpl:56M ep:63K epch:1571.68 loss:0.020 grdn:0.530 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 00:15:00 ot_train.py:423 step:110K smpl:56M ep:63K epch:1574.55 loss:0.020 grdn:0.534 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 00:15:00 ot_train.py:443 Checkpoint policy after step 110000
INFO 2026-08-11 00:16:11 ot_train.py:423 step:110K smpl:56M ep:63K epch:1577.41 loss:0.020 grdn:0.509 lr:5.0e-05 updt_s:0.329 data_s:0.025
INFO 2026-08-11 00:17:20 ot_train.py:423 step:110K smpl:57M ep:63K epch:1580.27 loss:0.020 grdn:inf lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 00:18:30 ot_train.py:423 step:111K smpl:57M ep:63K epch:1583.14 loss:0.021 grdn:0.549 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 00:19:41 ot_train.py:423 step:111K smpl:57M ep:63K epch:1586.00 loss:0.020 grdn:0.510 lr:5.0e-05 updt_s:0.329 data_s:0.025
INFO 2026-08-11 00:20:50 ot_train.py:423 step:111K smpl:57M ep:64K epch:1588.86 loss:0.020 grdn:0.546 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 00:21:59 ot_train.py:423 step:111K smpl:57M ep:64K epch:1591.72 loss:0.021 grdn:0.521 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 00:23:10 ot_train.py:423 step:111K smpl:57M ep:64K epch:1594.59 loss:0.020 grdn:0.510 lr:5.0e-05 updt_s:0.327 data_s:0.025
INFO 2026-08-11 00:24:18 ot_train.py:423 step:112K smpl:57M ep:64K epch:1597.45 loss:0.020 grdn:0.515 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 00:25:28 ot_train.py:423 step:112K smpl:57M ep:64K epch:1600.31 loss:0.022 grdn:0.538 lr:5.0e-05 updt_s:0.327 data_s:0.020
INFO 2026-08-11 00:26:39 ot_train.py:423 step:112K smpl:57M ep:64K epch:1603.18 loss:0.021 grdn:0.523 lr:5.0e-05 updt_s:0.330 data_s:0.025
INFO 2026-08-11 00:26:39 ot_train.py:443 Checkpoint policy after step 112000
INFO 2026-08-11 00:27:48 ot_train.py:423 step:112K smpl:57M ep:64K epch:1606.04 loss:0.020 grdn:0.507 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 00:28:59 ot_train.py:423 step:112K smpl:58M ep:64K epch:1608.90 loss:0.020 grdn:0.508 lr:5.0e-05 updt_s:0.328 data_s:0.025
INFO 2026-08-11 00:30:09 ot_train.py:423 step:113K smpl:58M ep:64K epch:1611.76 loss:0.020 grdn:0.533 lr:5.0e-05 updt_s:0.329 data_s:0.019
INFO 2026-08-11 00:31:18 ot_train.py:423 step:113K smpl:58M ep:65K epch:1614.63 loss:0.020 grdn:0.502 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 00:32:29 ot_train.py:423 step:113K smpl:58M ep:65K epch:1617.49 loss:0.020 grdn:0.524 lr:5.0e-05 updt_s:0.329 data_s:0.025
INFO 2026-08-11 00:33:38 ot_train.py:423 step:113K smpl:58M ep:65K epch:1620.35 loss:0.020 grdn:0.559 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 00:34:47 ot_train.py:423 step:113K smpl:58M ep:65K epch:1623.22 loss:0.020 grdn:0.532 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 00:35:58 ot_train.py:423 step:114K smpl:58M ep:65K epch:1626.08 loss:0.020 grdn:0.528 lr:5.0e-05 updt_s:0.330 data_s:0.025
INFO 2026-08-11 00:37:07 ot_train.py:423 step:114K smpl:58M ep:65K epch:1628.94 loss:0.020 grdn:0.527 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 00:38:17 ot_train.py:423 step:114K smpl:58M ep:65K epch:1631.80 loss:0.020 grdn:0.549 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 00:38:17 ot_train.py:443 Checkpoint policy after step 114000
INFO 2026-08-11 00:39:28 ot_train.py:423 step:114K smpl:58M ep:65K epch:1634.67 loss:0.020 grdn:0.530 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-11 00:40:37 ot_train.py:423 step:114K smpl:59M ep:66K epch:1637.53 loss:0.020 grdn:0.533 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 00:41:48 ot_train.py:423 step:115K smpl:59M ep:66K epch:1640.39 loss:0.020 grdn:inf lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-11 00:42:57 ot_train.py:423 step:115K smpl:59M ep:66K epch:1643.26 loss:0.020 grdn:0.533 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 00:44:07 ot_train.py:423 step:115K smpl:59M ep:66K epch:1646.12 loss:0.020 grdn:0.537 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 00:45:18 ot_train.py:423 step:115K smpl:59M ep:66K epch:1648.98 loss:0.020 grdn:0.515 lr:5.0e-05 updt_s:0.327 data_s:0.028
INFO 2026-08-11 00:46:27 ot_train.py:423 step:115K smpl:59M ep:66K epch:1651.84 loss:0.020 grdn:0.535 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 00:47:37 ot_train.py:423 step:116K smpl:59M ep:66K epch:1654.71 loss:0.020 grdn:0.518 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 00:48:47 ot_train.py:423 step:116K smpl:59M ep:66K epch:1657.57 loss:0.020 grdn:0.520 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-11 00:49:56 ot_train.py:423 step:116K smpl:59M ep:66K epch:1660.43 loss:0.022 grdn:0.598 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 00:49:56 ot_train.py:443 Checkpoint policy after step 116000
INFO 2026-08-11 00:51:06 ot_train.py:423 step:116K smpl:59M ep:67K epch:1663.30 loss:0.021 grdn:0.534 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 00:52:17 ot_train.py:423 step:116K smpl:60M ep:67K epch:1666.16 loss:0.020 grdn:0.534 lr:5.0e-05 updt_s:0.327 data_s:0.027
INFO 2026-08-11 00:53:26 ot_train.py:423 step:117K smpl:60M ep:67K epch:1669.02 loss:0.020 grdn:0.518 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 00:54:35 ot_train.py:423 step:117K smpl:60M ep:67K epch:1671.88 loss:0.020 grdn:inf lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 00:55:46 ot_train.py:423 step:117K smpl:60M ep:67K epch:1674.75 loss:0.020 grdn:0.494 lr:5.0e-05 updt_s:0.327 data_s:0.027
INFO 2026-08-11 00:56:55 ot_train.py:423 step:117K smpl:60M ep:67K epch:1677.61 loss:0.019 grdn:0.500 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 00:58:05 ot_train.py:423 step:117K smpl:60M ep:67K epch:1680.47 loss:0.020 grdn:0.526 lr:5.0e-05 updt_s:0.324 data_s:0.026
INFO 2026-08-11 00:59:14 ot_train.py:423 step:118K smpl:60M ep:67K epch:1683.33 loss:0.020 grdn:0.505 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 01:00:23 ot_train.py:423 step:118K smpl:60M ep:67K epch:1686.20 loss:0.020 grdn:0.525 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 01:01:34 ot_train.py:423 step:118K smpl:60M ep:68K epch:1689.06 loss:0.020 grdn:0.532 lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-11 01:01:34 ot_train.py:443 Checkpoint policy after step 118000
INFO 2026-08-11 01:02:43 ot_train.py:423 step:118K smpl:61M ep:68K epch:1691.92 loss:0.020 grdn:0.500 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 01:03:53 ot_train.py:423 step:118K smpl:61M ep:68K epch:1694.79 loss:0.020 grdn:0.524 lr:5.0e-05 updt_s:0.329 data_s:0.019
INFO 2026-08-11 01:05:04 ot_train.py:423 step:119K smpl:61M ep:68K epch:1697.65 loss:0.020 grdn:0.545 lr:5.0e-05 updt_s:0.324 data_s:0.027
INFO 2026-08-11 01:06:13 ot_train.py:423 step:119K smpl:61M ep:68K epch:1700.51 loss:0.020 grdn:inf lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 01:07:22 ot_train.py:423 step:119K smpl:61M ep:68K epch:1703.37 loss:0.020 grdn:0.532 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 01:08:33 ot_train.py:423 step:119K smpl:61M ep:68K epch:1706.24 loss:0.020 grdn:0.534 lr:5.0e-05 updt_s:0.327 data_s:0.027
INFO 2026-08-11 01:09:42 ot_train.py:423 step:119K smpl:61M ep:68K epch:1709.10 loss:0.020 grdn:0.533 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 01:10:51 ot_train.py:423 step:120K smpl:61M ep:68K epch:1711.96 loss:0.020 grdn:0.526 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 01:12:03 ot_train.py:423 step:120K smpl:61M ep:69K epch:1714.83 loss:0.020 grdn:0.539 lr:5.0e-05 updt_s:0.330 data_s:0.026
INFO 2026-08-11 01:13:12 ot_train.py:423 step:120K smpl:61M ep:69K epch:1717.69 loss:0.020 grdn:0.508 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 01:13:12 ot_train.py:443 Checkpoint policy after step 120000
INFO 2026-08-11 01:14:23 ot_train.py:423 step:120K smpl:62M ep:69K epch:1720.55 loss:0.019 grdn:0.539 lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-11 01:15:32 ot_train.py:423 step:120K smpl:62M ep:69K epch:1723.41 loss:0.020 grdn:0.533 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 01:16:41 ot_train.py:423 step:121K smpl:62M ep:69K epch:1726.28 loss:0.020 grdn:0.543 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 01:17:52 ot_train.py:423 step:121K smpl:62M ep:69K epch:1729.14 loss:0.020 grdn:inf lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-11 01:19:02 ot_train.py:423 step:121K smpl:62M ep:69K epch:1732.00 loss:0.020 grdn:0.517 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 01:20:11 ot_train.py:423 step:121K smpl:62M ep:69K epch:1734.87 loss:0.020 grdn:0.512 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 01:21:22 ot_train.py:423 step:121K smpl:62M ep:70K epch:1737.73 loss:0.020 grdn:0.515 lr:5.0e-05 updt_s:0.325 data_s:0.026
INFO 2026-08-11 01:22:31 ot_train.py:423 step:122K smpl:62M ep:70K epch:1740.59 loss:0.020 grdn:0.533 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 01:23:40 ot_train.py:423 step:122K smpl:62M ep:70K epch:1743.45 loss:0.020 grdn:0.512 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 01:24:51 ot_train.py:423 step:122K smpl:62M ep:70K epch:1746.32 loss:0.019 grdn:0.504 lr:5.0e-05 updt_s:0.328 data_s:0.025
INFO 2026-08-11 01:24:51 ot_train.py:443 Checkpoint policy after step 122000
INFO 2026-08-11 01:26:01 ot_train.py:423 step:122K smpl:63M ep:70K epch:1749.18 loss:0.019 grdn:0.511 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 01:27:10 ot_train.py:423 step:122K smpl:63M ep:70K epch:1752.04 loss:0.019 grdn:0.525 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 01:28:20 ot_train.py:423 step:123K smpl:63M ep:70K epch:1754.91 loss:0.020 grdn:0.503 lr:5.0e-05 updt_s:0.327 data_s:0.025
INFO 2026-08-11 01:29:29 ot_train.py:423 step:123K smpl:63M ep:70K epch:1757.77 loss:0.020 grdn:0.503 lr:5.0e-05 updt_s:0.323 data_s:0.019
INFO 2026-08-11 01:30:40 ot_train.py:423 step:123K smpl:63M ep:70K epch:1760.63 loss:0.020 grdn:inf lr:5.0e-05 updt_s:0.330 data_s:0.024
INFO 2026-08-11 01:31:49 ot_train.py:423 step:123K smpl:63M ep:71K epch:1763.49 loss:0.019 grdn:0.463 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 01:32:59 ot_train.py:423 step:123K smpl:63M ep:71K epch:1766.36 loss:0.020 grdn:0.547 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 01:34:09 ot_train.py:423 step:124K smpl:63M ep:71K epch:1769.22 loss:0.020 grdn:0.518 lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-11 01:35:18 ot_train.py:423 step:124K smpl:63M ep:71K epch:1772.08 loss:0.020 grdn:0.533 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 01:36:27 ot_train.py:423 step:124K smpl:63M ep:71K epch:1774.94 loss:0.019 grdn:0.517 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 01:36:27 ot_train.py:443 Checkpoint policy after step 124000
INFO 2026-08-11 01:37:39 ot_train.py:423 step:124K smpl:64M ep:71K epch:1777.81 loss:0.020 grdn:0.536 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-11 01:38:48 ot_train.py:423 step:124K smpl:64M ep:71K epch:1780.67 loss:0.020 grdn:0.507 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 01:39:57 ot_train.py:423 step:125K smpl:64M ep:71K epch:1783.53 loss:0.019 grdn:0.517 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 01:41:08 ot_train.py:423 step:125K smpl:64M ep:71K epch:1786.40 loss:0.020 grdn:0.528 lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-11 01:42:17 ot_train.py:423 step:125K smpl:64M ep:72K epch:1789.26 loss:0.019 grdn:inf lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 01:43:26 ot_train.py:423 step:125K smpl:64M ep:72K epch:1792.12 loss:0.020 grdn:0.543 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 01:44:37 ot_train.py:423 step:125K smpl:64M ep:72K epch:1794.98 loss:0.019 grdn:0.513 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-11 01:45:46 ot_train.py:423 step:126K smpl:64M ep:72K epch:1797.85 loss:0.019 grdn:0.531 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 01:46:57 ot_train.py:423 step:126K smpl:64M ep:72K epch:1800.71 loss:0.019 grdn:0.513 lr:5.0e-05 updt_s:0.329 data_s:0.025
INFO 2026-08-11 01:48:06 ot_train.py:423 step:126K smpl:65M ep:72K epch:1803.57 loss:0.020 grdn:0.527 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 01:48:06 ot_train.py:443 Checkpoint policy after step 126000
INFO 2026-08-11 01:49:16 ot_train.py:423 step:126K smpl:65M ep:72K epch:1806.44 loss:0.020 grdn:0.523 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 01:50:27 ot_train.py:423 step:126K smpl:65M ep:72K epch:1809.30 loss:0.020 grdn:0.517 lr:5.0e-05 updt_s:0.325 data_s:0.026
INFO 2026-08-11 01:51:36 ot_train.py:423 step:127K smpl:65M ep:72K epch:1812.16 loss:0.019 grdn:0.522 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 01:52:45 ot_train.py:423 step:127K smpl:65M ep:73K epch:1815.02 loss:0.019 grdn:0.521 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 01:53:56 ot_train.py:423 step:127K smpl:65M ep:73K epch:1817.89 loss:0.019 grdn:0.515 lr:5.0e-05 updt_s:0.330 data_s:0.024
INFO 2026-08-11 01:55:05 ot_train.py:423 step:127K smpl:65M ep:73K epch:1820.75 loss:0.019 grdn:inf lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 01:56:14 ot_train.py:423 step:127K smpl:65M ep:73K epch:1823.61 loss:0.019 grdn:0.499 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 01:57:25 ot_train.py:423 step:128K smpl:65M ep:73K epch:1826.48 loss:0.019 grdn:0.532 lr:5.0e-05 updt_s:0.328 data_s:0.024
INFO 2026-08-11 01:58:34 ot_train.py:423 step:128K smpl:65M ep:73K epch:1829.34 loss:0.019 grdn:0.499 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 01:59:43 ot_train.py:423 step:128K smpl:66M ep:73K epch:1832.20 loss:0.019 grdn:0.524 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 01:59:43 ot_train.py:443 Checkpoint policy after step 128000
INFO 2026-08-11 02:00:55 ot_train.py:423 step:128K smpl:66M ep:73K epch:1835.06 loss:0.019 grdn:0.527 lr:5.0e-05 updt_s:0.329 data_s:0.025
INFO 2026-08-11 02:02:04 ot_train.py:423 step:128K smpl:66M ep:74K epch:1837.93 loss:0.019 grdn:0.515 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 02:03:15 ot_train.py:423 step:129K smpl:66M ep:74K epch:1840.79 loss:0.020 grdn:0.532 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-11 02:04:24 ot_train.py:423 step:129K smpl:66M ep:74K epch:1843.65 loss:0.019 grdn:0.497 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 02:05:33 ot_train.py:423 step:129K smpl:66M ep:74K epch:1846.52 loss:0.019 grdn:0.522 lr:5.0e-05 updt_s:0.323 data_s:0.019
INFO 2026-08-11 02:06:43 ot_train.py:423 step:129K smpl:66M ep:74K epch:1849.38 loss:0.019 grdn:0.496 lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-11 02:07:53 ot_train.py:423 step:129K smpl:66M ep:74K epch:1852.24 loss:0.019 grdn:inf lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 02:09:02 ot_train.py:423 step:130K smpl:66M ep:74K epch:1855.10 loss:0.019 grdn:0.518 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 02:10:13 ot_train.py:423 step:130K smpl:66M ep:74K epch:1857.97 loss:0.019 grdn:0.526 lr:5.0e-05 updt_s:0.325 data_s:0.026
INFO 2026-08-11 02:11:22 ot_train.py:423 step:130K smpl:67M ep:74K epch:1860.83 loss:0.019 grdn:0.518 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 02:11:22 ot_train.py:443 Checkpoint policy after step 130000
INFO 2026-08-11 02:12:32 ot_train.py:423 step:130K smpl:67M ep:75K epch:1863.69 loss:0.019 grdn:0.534 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 02:13:43 ot_train.py:423 step:130K smpl:67M ep:75K epch:1866.55 loss:0.019 grdn:0.514 lr:5.0e-05 updt_s:0.329 data_s:0.026
INFO 2026-08-11 02:14:52 ot_train.py:423 step:131K smpl:67M ep:75K epch:1869.42 loss:0.019 grdn:0.524 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 02:16:01 ot_train.py:423 step:131K smpl:67M ep:75K epch:1872.28 loss:0.019 grdn:0.489 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 02:17:12 ot_train.py:423 step:131K smpl:67M ep:75K epch:1875.14 loss:0.019 grdn:0.526 lr:5.0e-05 updt_s:0.329 data_s:0.025
INFO 2026-08-11 02:18:21 ot_train.py:423 step:131K smpl:67M ep:75K epch:1878.01 loss:0.019 grdn:0.525 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 02:19:32 ot_train.py:423 step:131K smpl:67M ep:75K epch:1880.87 loss:0.019 grdn:inf lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-11 02:20:42 ot_train.py:423 step:132K smpl:67M ep:75K epch:1883.73 loss:0.019 grdn:0.519 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 02:21:50 ot_train.py:423 step:132K smpl:67M ep:75K epch:1886.59 loss:0.019 grdn:0.512 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 02:23:01 ot_train.py:423 step:132K smpl:68M ep:76K epch:1889.46 loss:0.019 grdn:0.504 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-11 02:23:01 ot_train.py:443 Checkpoint policy after step 132000
INFO 2026-08-11 02:24:12 ot_train.py:423 step:132K smpl:68M ep:76K epch:1892.32 loss:0.019 grdn:0.524 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 02:25:20 ot_train.py:423 step:132K smpl:68M ep:76K epch:1895.18 loss:0.019 grdn:0.500 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 02:26:31 ot_train.py:423 step:133K smpl:68M ep:76K epch:1898.05 loss:0.019 grdn:0.538 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-11 02:27:41 ot_train.py:423 step:133K smpl:68M ep:76K epch:1900.91 loss:0.019 grdn:0.544 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 02:28:50 ot_train.py:423 step:133K smpl:68M ep:76K epch:1903.77 loss:0.019 grdn:0.497 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 02:30:00 ot_train.py:423 step:133K smpl:68M ep:76K epch:1906.63 loss:0.019 grdn:0.507 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-11 02:31:10 ot_train.py:423 step:133K smpl:68M ep:76K epch:1909.50 loss:0.019 grdn:inf lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 02:32:19 ot_train.py:423 step:134K smpl:68M ep:76K epch:1912.36 loss:0.019 grdn:0.480 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 02:33:30 ot_train.py:423 step:134K smpl:69M ep:77K epch:1915.22 loss:0.019 grdn:0.509 lr:5.0e-05 updt_s:0.329 data_s:0.025
INFO 2026-08-11 02:34:38 ot_train.py:423 step:134K smpl:69M ep:77K epch:1918.09 loss:0.019 grdn:0.501 lr:5.0e-05 updt_s:0.322 data_s:0.019
INFO 2026-08-11 02:34:38 ot_train.py:443 Checkpoint policy after step 134000
INFO 2026-08-11 02:35:50 ot_train.py:423 step:134K smpl:69M ep:77K epch:1920.95 loss:0.019 grdn:0.518 lr:5.0e-05 updt_s:0.329 data_s:0.026
INFO 2026-08-11 02:36:59 ot_train.py:423 step:134K smpl:69M ep:77K epch:1923.81 loss:0.019 grdn:0.507 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 02:38:09 ot_train.py:423 step:135K smpl:69M ep:77K epch:1926.67 loss:0.019 grdn:0.492 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 02:39:19 ot_train.py:423 step:135K smpl:69M ep:77K epch:1929.54 loss:0.019 grdn:0.510 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-11 02:40:28 ot_train.py:423 step:135K smpl:69M ep:77K epch:1932.40 loss:0.019 grdn:0.512 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 02:41:38 ot_train.py:423 step:135K smpl:69M ep:77K epch:1935.26 loss:0.019 grdn:0.516 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 02:42:49 ot_train.py:423 step:135K smpl:69M ep:78K epch:1938.13 loss:0.019 grdn:0.565 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-11 02:43:58 ot_train.py:423 step:136K smpl:69M ep:78K epch:1940.99 loss:0.019 grdn:inf lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 02:45:06 ot_train.py:423 step:136K smpl:70M ep:78K epch:1943.85 loss:0.019 grdn:0.513 lr:5.0e-05 updt_s:0.322 data_s:0.019
INFO 2026-08-11 02:46:17 ot_train.py:423 step:136K smpl:70M ep:78K epch:1946.71 loss:0.019 grdn:0.530 lr:5.0e-05 updt_s:0.328 data_s:0.024
INFO 2026-08-11 02:46:17 ot_train.py:443 Checkpoint policy after step 136000
INFO 2026-08-11 02:47:27 ot_train.py:423 step:136K smpl:70M ep:78K epch:1949.58 loss:0.019 grdn:0.484 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 02:48:37 ot_train.py:423 step:136K smpl:70M ep:78K epch:1952.44 loss:0.019 grdn:0.517 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-11 02:49:47 ot_train.py:423 step:137K smpl:70M ep:78K epch:1955.30 loss:0.019 grdn:0.525 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 02:50:56 ot_train.py:423 step:137K smpl:70M ep:78K epch:1958.16 loss:0.019 grdn:0.520 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 02:52:06 ot_train.py:423 step:137K smpl:70M ep:78K epch:1961.03 loss:0.019 grdn:0.519 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-11 02:53:15 ot_train.py:423 step:137K smpl:70M ep:79K epch:1963.89 loss:0.019 grdn:0.526 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 02:54:24 ot_train.py:423 step:137K smpl:70M ep:79K epch:1966.75 loss:0.019 grdn:0.490 lr:5.0e-05 updt_s:0.323 data_s:0.019
INFO 2026-08-11 02:55:35 ot_train.py:423 step:138K smpl:70M ep:79K epch:1969.62 loss:0.018 grdn:0.498 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-11 02:56:44 ot_train.py:423 step:138K smpl:71M ep:79K epch:1972.48 loss:0.019 grdn:inf lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 02:57:53 ot_train.py:423 step:138K smpl:71M ep:79K epch:1975.34 loss:0.019 grdn:0.518 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 02:57:53 ot_train.py:443 Checkpoint policy after step 138000
INFO 2026-08-11 02:59:04 ot_train.py:423 step:138K smpl:71M ep:79K epch:1978.20 loss:0.019 grdn:0.530 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-11 03:00:13 ot_train.py:423 step:138K smpl:71M ep:79K epch:1981.07 loss:0.018 grdn:0.512 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 03:01:22 ot_train.py:423 step:139K smpl:71M ep:79K epch:1983.93 loss:0.019 grdn:0.511 lr:5.0e-05 updt_s:0.323 data_s:0.019
INFO 2026-08-11 03:02:33 ot_train.py:423 step:139K smpl:71M ep:79K epch:1986.79 loss:0.019 grdn:0.519 lr:5.0e-05 updt_s:0.329 data_s:0.025
INFO 2026-08-11 03:03:42 ot_train.py:423 step:139K smpl:71M ep:80K epch:1989.66 loss:0.019 grdn:0.478 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 03:04:53 ot_train.py:423 step:139K smpl:71M ep:80K epch:1992.52 loss:0.019 grdn:0.502 lr:5.0e-05 updt_s:0.328 data_s:0.025
INFO 2026-08-11 03:06:02 ot_train.py:423 step:139K smpl:71M ep:80K epch:1995.38 loss:0.019 grdn:0.512 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 03:07:11 ot_train.py:423 step:140K smpl:71M ep:80K epch:1998.24 loss:0.019 grdn:0.507 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 03:08:22 ot_train.py:423 step:140K smpl:72M ep:80K epch:2001.11 loss:0.019 grdn:0.509 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-11 03:09:31 ot_train.py:423 step:140K smpl:72M ep:80K epch:2003.97 loss:0.019 grdn:inf lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 03:09:31 ot_train.py:443 Checkpoint policy after step 140000
INFO 2026-08-11 03:10:41 ot_train.py:423 step:140K smpl:72M ep:80K epch:2006.83 loss:0.019 grdn:0.511 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 03:11:52 ot_train.py:423 step:140K smpl:72M ep:80K epch:2009.70 loss:0.018 grdn:0.501 lr:5.0e-05 updt_s:0.328 data_s:0.024
INFO 2026-08-11 03:13:01 ot_train.py:423 step:141K smpl:72M ep:81K epch:2012.56 loss:0.019 grdn:0.511 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 03:14:10 ot_train.py:423 step:141K smpl:72M ep:81K epch:2015.42 loss:0.019 grdn:0.533 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 03:15:21 ot_train.py:423 step:141K smpl:72M ep:81K epch:2018.28 loss:0.019 grdn:0.508 lr:5.0e-05 updt_s:0.329 data_s:0.026
INFO 2026-08-11 03:16:31 ot_train.py:423 step:141K smpl:72M ep:81K epch:2021.15 loss:0.019 grdn:0.523 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 03:17:39 ot_train.py:423 step:141K smpl:72M ep:81K epch:2024.01 loss:0.018 grdn:0.516 lr:5.0e-05 updt_s:0.323 data_s:0.019
INFO 2026-08-11 03:18:50 ot_train.py:423 step:142K smpl:72M ep:81K epch:2026.87 loss:0.019 grdn:0.515 lr:5.0e-05 updt_s:0.328 data_s:0.025
INFO 2026-08-11 03:20:00 ot_train.py:423 step:142K smpl:73M ep:81K epch:2029.74 loss:0.019 grdn:0.546 lr:5.0e-05 updt_s:0.329 data_s:0.019
INFO 2026-08-11 03:21:10 ot_train.py:423 step:142K smpl:73M ep:81K epch:2032.60 loss:0.019 grdn:0.507 lr:5.0e-05 updt_s:0.326 data_s:0.025
INFO 2026-08-11 03:21:10 ot_train.py:443 Checkpoint policy after step 142000
INFO 2026-08-11 03:22:20 ot_train.py:423 step:142K smpl:73M ep:81K epch:2035.46 loss:0.019 grdn:inf lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 03:23:29 ot_train.py:423 step:142K smpl:73M ep:82K epch:2038.32 loss:0.018 grdn:0.511 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 03:24:40 ot_train.py:423 step:143K smpl:73M ep:82K epch:2041.19 loss:0.018 grdn:0.515 lr:5.0e-05 updt_s:0.329 data_s:0.026
INFO 2026-08-11 03:25:49 ot_train.py:423 step:143K smpl:73M ep:82K epch:2044.05 loss:0.018 grdn:0.501 lr:5.0e-05 updt_s:0.323 data_s:0.019
INFO 2026-08-11 03:26:58 ot_train.py:423 step:143K smpl:73M ep:82K epch:2046.91 loss:0.019 grdn:0.519 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 03:28:09 ot_train.py:423 step:143K smpl:73M ep:82K epch:2049.77 loss:0.018 grdn:0.513 lr:5.0e-05 updt_s:0.328 data_s:0.025
INFO 2026-08-11 03:29:18 ot_train.py:423 step:143K smpl:73M ep:82K epch:2052.64 loss:0.018 grdn:0.514 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 03:30:27 ot_train.py:423 step:144K smpl:74M ep:82K epch:2055.50 loss:0.018 grdn:0.529 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 03:31:38 ot_train.py:423 step:144K smpl:74M ep:82K epch:2058.36 loss:0.018 grdn:0.494 lr:5.0e-05 updt_s:0.327 data_s:0.027
INFO 2026-08-11 03:32:48 ot_train.py:423 step:144K smpl:74M ep:82K epch:2061.23 loss:0.018 grdn:0.526 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 03:32:48 ot_train.py:443 Checkpoint policy after step 144000
INFO 2026-08-11 03:33:58 ot_train.py:423 step:144K smpl:74M ep:83K epch:2064.09 loss:0.019 grdn:inf lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 03:35:08 ot_train.py:423 step:144K smpl:74M ep:83K epch:2066.95 loss:0.018 grdn:0.525 lr:5.0e-05 updt_s:0.329 data_s:0.025
INFO 2026-08-11 03:36:18 ot_train.py:423 step:145K smpl:74M ep:83K epch:2069.81 loss:0.018 grdn:0.514 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 03:37:29 ot_train.py:423 step:145K smpl:74M ep:83K epch:2072.68 loss:0.019 grdn:0.500 lr:5.0e-05 updt_s:0.329 data_s:0.024
INFO 2026-08-11 03:38:38 ot_train.py:423 step:145K smpl:74M ep:83K epch:2075.54 loss:0.019 grdn:0.542 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 03:39:47 ot_train.py:423 step:145K smpl:74M ep:83K epch:2078.40 loss:0.018 grdn:0.507 lr:5.0e-05 updt_s:0.322 data_s:0.019
INFO 2026-08-11 03:40:58 ot_train.py:423 step:145K smpl:74M ep:83K epch:2081.27 loss:0.018 grdn:0.515 lr:5.0e-05 updt_s:0.330 data_s:0.026
INFO 2026-08-11 03:42:07 ot_train.py:423 step:146K smpl:75M ep:83K epch:2084.13 loss:0.018 grdn:0.516 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 03:43:16 ot_train.py:423 step:146K smpl:75M ep:83K epch:2086.99 loss:0.018 grdn:0.491 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 03:44:27 ot_train.py:423 step:146K smpl:75M ep:84K epch:2089.85 loss:0.019 grdn:0.522 lr:5.0e-05 updt_s:0.328 data_s:0.025
INFO 2026-08-11 03:44:27 ot_train.py:443 Checkpoint policy after step 146000
INFO 2026-08-11 03:45:37 ot_train.py:423 step:146K smpl:75M ep:84K epch:2092.72 loss:0.018 grdn:inf lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 03:46:46 ot_train.py:423 step:146K smpl:75M ep:84K epch:2095.58 loss:0.019 grdn:0.527 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 03:47:57 ot_train.py:423 step:147K smpl:75M ep:84K epch:2098.44 loss:0.018 grdn:0.518 lr:5.0e-05 updt_s:0.330 data_s:0.026
INFO 2026-08-11 03:49:06 ot_train.py:423 step:147K smpl:75M ep:84K epch:2101.31 loss:0.018 grdn:0.478 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 03:50:14 ot_train.py:423 step:147K smpl:75M ep:84K epch:2104.17 loss:0.018 grdn:0.536 lr:5.0e-05 updt_s:0.322 data_s:0.019
INFO 2026-08-11 03:51:25 ot_train.py:423 step:147K smpl:75M ep:84K epch:2107.03 loss:0.018 grdn:0.540 lr:5.0e-05 updt_s:0.327 data_s:0.027
INFO 2026-08-11 03:52:35 ot_train.py:423 step:147K smpl:75M ep:84K epch:2109.89 loss:0.018 grdn:0.539 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 03:53:45 ot_train.py:423 step:148K smpl:76M ep:85K epch:2112.76 loss:0.018 grdn:0.517 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-11 03:54:55 ot_train.py:423 step:148K smpl:76M ep:85K epch:2115.62 loss:0.018 grdn:0.519 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 03:56:04 ot_train.py:423 step:148K smpl:76M ep:85K epch:2118.48 loss:0.018 grdn:0.518 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 03:56:04 ot_train.py:443 Checkpoint policy after step 148000
INFO 2026-08-11 03:57:15 ot_train.py:423 step:148K smpl:76M ep:85K epch:2121.35 loss:0.018 grdn:0.506 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-11 03:58:24 ot_train.py:423 step:148K smpl:76M ep:85K epch:2124.21 loss:0.018 grdn:inf lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 03:59:33 ot_train.py:423 step:149K smpl:76M ep:85K epch:2127.07 loss:0.018 grdn:0.525 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 04:00:44 ot_train.py:423 step:149K smpl:76M ep:85K epch:2129.93 loss:0.018 grdn:0.515 lr:5.0e-05 updt_s:0.329 data_s:0.026
INFO 2026-08-11 04:01:53 ot_train.py:423 step:149K smpl:76M ep:85K epch:2132.80 loss:0.018 grdn:0.558 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 04:03:02 ot_train.py:423 step:149K smpl:76M ep:85K epch:2135.66 loss:0.020 grdn:0.543 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 04:04:13 ot_train.py:423 step:149K smpl:76M ep:86K epch:2138.52 loss:0.018 grdn:0.519 lr:5.0e-05 updt_s:0.330 data_s:0.025
INFO 2026-08-11 04:05:22 ot_train.py:423 step:150K smpl:77M ep:86K epch:2141.38 loss:0.018 grdn:0.504 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 04:06:31 ot_train.py:423 step:150K smpl:77M ep:86K epch:2144.25 loss:0.018 grdn:0.496 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 04:07:42 ot_train.py:423 step:150K smpl:77M ep:86K epch:2147.11 loss:0.018 grdn:0.500 lr:5.0e-05 updt_s:0.329 data_s:0.025
INFO 2026-08-11 04:07:42 ot_train.py:443 Checkpoint policy after step 150000
INFO 2026-08-11 04:08:52 ot_train.py:423 step:150K smpl:77M ep:86K epch:2149.97 loss:0.018 grdn:0.511 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 04:10:02 ot_train.py:423 step:150K smpl:77M ep:86K epch:2152.84 loss:0.018 grdn:inf lr:5.0e-05 updt_s:0.327 data_s:0.025
INFO 2026-08-11 04:11:12 ot_train.py:423 step:151K smpl:77M ep:86K epch:2155.70 loss:0.018 grdn:0.507 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 04:12:21 ot_train.py:423 step:151K smpl:77M ep:86K epch:2158.56 loss:0.018 grdn:0.508 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 04:13:31 ot_train.py:423 step:151K smpl:77M ep:86K epch:2161.42 loss:0.018 grdn:0.510 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-11 04:14:40 ot_train.py:423 step:151K smpl:77M ep:87K epch:2164.29 loss:0.018 grdn:0.515 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 04:15:49 ot_train.py:423 step:151K smpl:78M ep:87K epch:2167.15 loss:0.018 grdn:0.525 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 04:17:00 ot_train.py:423 step:152K smpl:78M ep:87K epch:2170.01 loss:0.018 grdn:0.540 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-11 04:18:08 ot_train.py:423 step:152K smpl:78M ep:87K epch:2172.88 loss:0.018 grdn:0.512 lr:5.0e-05 updt_s:0.322 data_s:0.019
INFO 2026-08-11 04:19:17 ot_train.py:423 step:152K smpl:78M ep:87K epch:2175.74 loss:0.018 grdn:0.548 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 04:19:17 ot_train.py:443 Checkpoint policy after step 152000
INFO 2026-08-11 04:20:29 ot_train.py:423 step:152K smpl:78M ep:87K epch:2178.60 loss:0.018 grdn:0.475 lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-11 04:21:38 ot_train.py:423 step:152K smpl:78M ep:87K epch:2181.46 loss:0.018 grdn:0.498 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 04:22:47 ot_train.py:423 step:153K smpl:78M ep:87K epch:2184.33 loss:0.018 grdn:inf lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 04:23:57 ot_train.py:423 step:153K smpl:78M ep:87K epch:2187.19 loss:0.018 grdn:0.488 lr:5.0e-05 updt_s:0.326 data_s:0.025
INFO 2026-08-11 04:25:07 ot_train.py:423 step:153K smpl:78M ep:88K epch:2190.05 loss:0.018 grdn:0.513 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 04:26:17 ot_train.py:423 step:153K smpl:78M ep:88K epch:2192.92 loss:0.018 grdn:0.521 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-11 04:27:27 ot_train.py:423 step:153K smpl:79M ep:88K epch:2195.78 loss:0.018 grdn:0.509 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 04:28:35 ot_train.py:423 step:154K smpl:79M ep:88K epch:2198.64 loss:0.018 grdn:0.512 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 04:29:46 ot_train.py:423 step:154K smpl:79M ep:88K epch:2201.50 loss:0.018 grdn:0.509 lr:5.0e-05 updt_s:0.329 data_s:0.026
INFO 2026-08-11 04:30:55 ot_train.py:423 step:154K smpl:79M ep:88K epch:2204.37 loss:0.018 grdn:0.521 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 04:30:55 ot_train.py:443 Checkpoint policy after step 154000
INFO 2026-08-11 04:32:05 ot_train.py:423 step:154K smpl:79M ep:88K epch:2207.23 loss:0.018 grdn:0.492 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 04:33:16 ot_train.py:423 step:154K smpl:79M ep:88K epch:2210.09 loss:0.018 grdn:0.504 lr:5.0e-05 updt_s:0.330 data_s:0.025
INFO 2026-08-11 04:34:25 ot_train.py:423 step:155K smpl:79M ep:89K epch:2212.96 loss:0.018 grdn:inf lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 04:35:34 ot_train.py:423 step:155K smpl:79M ep:89K epch:2215.82 loss:0.018 grdn:0.474 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 04:36:45 ot_train.py:423 step:155K smpl:79M ep:89K epch:2218.68 loss:0.018 grdn:0.504 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-11 04:37:54 ot_train.py:423 step:155K smpl:79M ep:89K epch:2221.54 loss:0.018 grdn:0.515 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 04:39:04 ot_train.py:423 step:155K smpl:80M ep:89K epch:2224.41 loss:0.018 grdn:0.531 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 04:40:14 ot_train.py:423 step:156K smpl:80M ep:89K epch:2227.27 loss:0.018 grdn:0.509 lr:5.0e-05 updt_s:0.327 data_s:0.025
INFO 2026-08-11 04:41:23 ot_train.py:423 step:156K smpl:80M ep:89K epch:2230.13 loss:0.018 grdn:0.516 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 04:42:34 ot_train.py:423 step:156K smpl:80M ep:89K epch:2233.00 loss:0.018 grdn:0.526 lr:5.0e-05 updt_s:0.327 data_s:0.025
INFO 2026-08-11 04:42:34 ot_train.py:443 Checkpoint policy after step 156000
INFO 2026-08-11 04:43:44 ot_train.py:423 step:156K smpl:80M ep:89K epch:2235.86 loss:0.018 grdn:0.535 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 04:44:53 ot_train.py:423 step:156K smpl:80M ep:90K epch:2238.72 loss:0.018 grdn:0.516 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 04:46:04 ot_train.py:423 step:157K smpl:80M ep:90K epch:2241.58 loss:0.018 grdn:inf lr:5.0e-05 updt_s:0.326 data_s:0.027
INFO 2026-08-11 04:47:13 ot_train.py:423 step:157K smpl:80M ep:90K epch:2244.45 loss:0.018 grdn:0.508 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 04:48:23 ot_train.py:423 step:157K smpl:80M ep:90K epch:2247.31 loss:0.018 grdn:0.482 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 04:49:34 ot_train.py:423 step:157K smpl:80M ep:90K epch:2250.17 loss:0.018 grdn:0.507 lr:5.0e-05 updt_s:0.329 data_s:0.026
INFO 2026-08-11 04:50:42 ot_train.py:423 step:157K smpl:81M ep:90K epch:2253.03 loss:0.018 grdn:0.564 lr:5.0e-05 updt_s:0.323 data_s:0.019
INFO 2026-08-11 04:51:52 ot_train.py:423 step:158K smpl:81M ep:90K epch:2255.90 loss:0.018 grdn:0.505 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 04:53:03 ot_train.py:423 step:158K smpl:81M ep:90K epch:2258.76 loss:0.018 grdn:0.482 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-11 04:54:12 ot_train.py:423 step:158K smpl:81M ep:90K epch:2261.62 loss:0.018 grdn:0.482 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 04:54:12 ot_train.py:443 Checkpoint policy after step 158000
INFO 2026-08-11 04:55:23 ot_train.py:423 step:158K smpl:81M ep:91K epch:2264.49 loss:0.018 grdn:0.505 lr:5.0e-05 updt_s:0.330 data_s:0.024
INFO 2026-08-11 04:56:33 ot_train.py:423 step:158K smpl:81M ep:91K epch:2267.35 loss:0.018 grdn:0.512 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 04:57:42 ot_train.py:423 step:159K smpl:81M ep:91K epch:2270.21 loss:0.018 grdn:0.550 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 04:58:52 ot_train.py:423 step:159K smpl:81M ep:91K epch:2273.07 loss:0.018 grdn:inf lr:5.0e-05 updt_s:0.326 data_s:0.027
INFO 2026-08-11 05:00:02 ot_train.py:423 step:159K smpl:81M ep:91K epch:2275.94 loss:0.017 grdn:0.515 lr:5.0e-05 updt_s:0.329 data_s:0.019
INFO 2026-08-11 05:01:11 ot_train.py:423 step:159K smpl:82M ep:91K epch:2278.80 loss:0.018 grdn:0.519 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 05:02:22 ot_train.py:423 step:159K smpl:82M ep:91K epch:2281.66 loss:0.018 grdn:0.474 lr:5.0e-05 updt_s:0.330 data_s:0.025
INFO 2026-08-11 05:03:32 ot_train.py:423 step:160K smpl:82M ep:91K epch:2284.53 loss:0.018 grdn:0.551 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 05:04:40 ot_train.py:423 step:160K smpl:82M ep:91K epch:2287.39 loss:0.018 grdn:0.487 lr:5.0e-05 updt_s:0.323 data_s:0.019
INFO 2026-08-11 05:05:51 ot_train.py:423 step:160K smpl:82M ep:92K epch:2290.25 loss:0.018 grdn:0.501 lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-11 05:05:51 ot_train.py:443 Checkpoint policy after step 160000
INFO 2026-08-11 05:07:01 ot_train.py:423 step:160K smpl:82M ep:92K epch:2293.11 loss:0.018 grdn:0.504 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 05:08:10 ot_train.py:423 step:160K smpl:82M ep:92K epch:2295.98 loss:0.018 grdn:0.523 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 05:09:21 ot_train.py:423 step:161K smpl:82M ep:92K epch:2298.84 loss:0.017 grdn:0.496 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-11 05:10:30 ot_train.py:423 step:161K smpl:82M ep:92K epch:2301.70 loss:0.018 grdn:inf lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 05:11:41 ot_train.py:423 step:161K smpl:82M ep:92K epch:2304.57 loss:0.018 grdn:0.514 lr:5.0e-05 updt_s:0.327 data_s:0.024
INFO 2026-08-11 05:12:51 ot_train.py:423 step:161K smpl:83M ep:92K epch:2307.43 loss:0.018 grdn:0.485 lr:5.0e-05 updt_s:0.329 data_s:0.019
INFO 2026-08-11 05:13:59 ot_train.py:423 step:161K smpl:83M ep:92K epch:2310.29 loss:0.017 grdn:0.513 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 05:15:10 ot_train.py:423 step:162K smpl:83M ep:93K epch:2313.15 loss:0.017 grdn:0.491 lr:5.0e-05 updt_s:0.328 data_s:0.024
INFO 2026-08-11 05:16:20 ot_train.py:423 step:162K smpl:83M ep:93K epch:2316.02 loss:0.018 grdn:0.501 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 05:17:29 ot_train.py:423 step:162K smpl:83M ep:93K epch:2318.88 loss:0.017 grdn:0.535 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 05:17:29 ot_train.py:443 Checkpoint policy after step 162000
INFO 2026-08-11 05:18:40 ot_train.py:423 step:162K smpl:83M ep:93K epch:2321.74 loss:0.018 grdn:0.529 lr:5.0e-05 updt_s:0.327 data_s:0.025
INFO 2026-08-11 05:19:49 ot_train.py:423 step:162K smpl:83M ep:93K epch:2324.61 loss:0.018 grdn:0.514 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 05:20:58 ot_train.py:423 step:163K smpl:83M ep:93K epch:2327.47 loss:0.018 grdn:0.532 lr:5.0e-05 updt_s:0.323 data_s:0.019
INFO 2026-08-11 05:22:09 ot_train.py:423 step:163K smpl:83M ep:93K epch:2330.33 loss:0.017 grdn:0.496 lr:5.0e-05 updt_s:0.330 data_s:0.025
INFO 2026-08-11 05:23:18 ot_train.py:423 step:163K smpl:83M ep:93K epch:2333.19 loss:0.018 grdn:inf lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 05:24:27 ot_train.py:423 step:163K smpl:84M ep:93K epch:2336.06 loss:0.018 grdn:0.538 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 05:25:38 ot_train.py:423 step:163K smpl:84M ep:94K epch:2338.92 loss:0.017 grdn:0.488 lr:5.0e-05 updt_s:0.326 data_s:0.027
INFO 2026-08-11 05:26:47 ot_train.py:423 step:164K smpl:84M ep:94K epch:2341.78 loss:0.018 grdn:0.488 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 05:27:58 ot_train.py:423 step:164K smpl:84M ep:94K epch:2344.64 loss:0.017 grdn:0.518 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-11 05:29:07 ot_train.py:423 step:164K smpl:84M ep:94K epch:2347.51 loss:0.018 grdn:0.512 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 05:29:07 ot_train.py:443 Checkpoint policy after step 164000
INFO 2026-08-11 05:30:17 ot_train.py:423 step:164K smpl:84M ep:94K epch:2350.37 loss:0.018 grdn:0.526 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 05:31:28 ot_train.py:423 step:164K smpl:84M ep:94K epch:2353.23 loss:0.018 grdn:0.488 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-11 05:32:37 ot_train.py:423 step:165K smpl:84M ep:94K epch:2356.10 loss:0.018 grdn:0.517 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 05:33:46 ot_train.py:423 step:165K smpl:84M ep:94K epch:2358.96 loss:0.017 grdn:0.537 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 05:34:56 ot_train.py:423 step:165K smpl:84M ep:94K epch:2361.82 loss:0.017 grdn:inf lr:5.0e-05 updt_s:0.326 data_s:0.025
INFO 2026-08-11 05:36:06 ot_train.py:423 step:165K smpl:85M ep:95K epch:2364.68 loss:0.017 grdn:0.533 lr:5.0e-05 updt_s:0.329 data_s:0.019
INFO 2026-08-11 05:37:15 ot_train.py:423 step:165K smpl:85M ep:95K epch:2367.55 loss:0.018 grdn:0.497 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 05:38:26 ot_train.py:423 step:166K smpl:85M ep:95K epch:2370.41 loss:0.018 grdn:0.527 lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-11 05:39:35 ot_train.py:423 step:166K smpl:85M ep:95K epch:2373.27 loss:0.018 grdn:0.517 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 05:40:44 ot_train.py:423 step:166K smpl:85M ep:95K epch:2376.14 loss:0.017 grdn:0.489 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 05:40:44 ot_train.py:443 Checkpoint policy after step 166000
INFO 2026-08-11 05:41:56 ot_train.py:423 step:166K smpl:85M ep:95K epch:2379.00 loss:0.017 grdn:0.520 lr:5.0e-05 updt_s:0.329 data_s:0.026
INFO 2026-08-11 05:43:05 ot_train.py:423 step:166K smpl:85M ep:95K epch:2381.86 loss:0.018 grdn:0.499 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 05:44:15 ot_train.py:423 step:167K smpl:85M ep:95K epch:2384.72 loss:0.018 grdn:0.531 lr:5.0e-05 updt_s:0.326 data_s:0.027
INFO 2026-08-11 05:45:25 ot_train.py:423 step:167K smpl:85M ep:96K epch:2387.59 loss:0.018 grdn:0.527 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 05:46:34 ot_train.py:423 step:167K smpl:86M ep:96K epch:2390.45 loss:0.018 grdn:inf lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 05:47:45 ot_train.py:423 step:167K smpl:86M ep:96K epch:2393.31 loss:0.017 grdn:0.493 lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-11 05:48:54 ot_train.py:423 step:167K smpl:86M ep:96K epch:2396.18 loss:0.017 grdn:0.502 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 05:50:03 ot_train.py:423 step:168K smpl:86M ep:96K epch:2399.04 loss:0.017 grdn:0.473 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 05:51:14 ot_train.py:423 step:168K smpl:86M ep:96K epch:2401.90 loss:0.018 grdn:0.539 lr:5.0e-05 updt_s:0.329 data_s:0.025
INFO 2026-08-11 05:52:23 ot_train.py:423 step:168K smpl:86M ep:96K epch:2404.76 loss:0.018 grdn:0.508 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 05:52:23 ot_train.py:443 Checkpoint policy after step 168000
INFO 2026-08-11 05:53:32 ot_train.py:423 step:168K smpl:86M ep:96K epch:2407.63 loss:0.017 grdn:0.461 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 05:54:43 ot_train.py:423 step:168K smpl:86M ep:96K epch:2410.49 loss:0.017 grdn:0.546 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-11 05:55:52 ot_train.py:423 step:169K smpl:86M ep:97K epch:2413.35 loss:0.018 grdn:0.518 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 05:57:01 ot_train.py:423 step:169K smpl:86M ep:97K epch:2416.22 loss:0.017 grdn:0.511 lr:5.0e-05 updt_s:0.323 data_s:0.019
INFO 2026-08-11 05:58:11 ot_train.py:423 step:169K smpl:87M ep:97K epch:2419.08 loss:0.017 grdn:0.513 lr:5.0e-05 updt_s:0.328 data_s:0.025
INFO 2026-08-11 05:59:21 ot_train.py:423 step:169K smpl:87M ep:97K epch:2421.94 loss:0.017 grdn:inf lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 06:00:32 ot_train.py:423 step:169K smpl:87M ep:97K epch:2424.80 loss:0.017 grdn:0.526 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-11 06:01:41 ot_train.py:423 step:170K smpl:87M ep:97K epch:2427.67 loss:0.017 grdn:0.495 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 06:02:50 ot_train.py:423 step:170K smpl:87M ep:97K epch:2430.53 loss:0.017 grdn:0.526 lr:5.0e-05 updt_s:0.323 data_s:0.019
INFO 2026-08-11 06:04:01 ot_train.py:423 step:170K smpl:87M ep:97K epch:2433.39 loss:0.017 grdn:0.508 lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-11 06:04:01 ot_train.py:443 Checkpoint policy after step 170000
INFO 2026-08-11 06:05:11 ot_train.py:423 step:170K smpl:87M ep:97K epch:2436.25 loss:0.017 grdn:0.518 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 06:06:20 ot_train.py:423 step:170K smpl:87M ep:98K epch:2439.12 loss:0.017 grdn:0.485 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 06:07:31 ot_train.py:423 step:171K smpl:87M ep:98K epch:2441.98 loss:0.017 grdn:0.534 lr:5.0e-05 updt_s:0.327 data_s:0.024
INFO 2026-08-11 06:08:40 ot_train.py:423 step:171K smpl:87M ep:98K epch:2444.84 loss:0.017 grdn:0.542 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 06:09:49 ot_train.py:423 step:171K smpl:88M ep:98K epch:2447.71 loss:0.017 grdn:0.549 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 06:11:00 ot_train.py:423 step:171K smpl:88M ep:98K epch:2450.57 loss:0.018 grdn:0.490 lr:5.0e-05 updt_s:0.328 data_s:0.025
INFO 2026-08-11 06:12:09 ot_train.py:423 step:171K smpl:88M ep:98K epch:2453.43 loss:0.017 grdn:inf lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 06:13:18 ot_train.py:423 step:172K smpl:88M ep:98K epch:2456.29 loss:0.017 grdn:0.490 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 06:14:29 ot_train.py:423 step:172K smpl:88M ep:98K epch:2459.16 loss:0.017 grdn:0.489 lr:5.0e-05 updt_s:0.331 data_s:0.026
INFO 2026-08-11 06:15:39 ot_train.py:423 step:172K smpl:88M ep:98K epch:2462.02 loss:0.017 grdn:0.519 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 06:15:39 ot_train.py:443 Checkpoint policy after step 172000
INFO 2026-08-11 06:16:50 ot_train.py:423 step:172K smpl:88M ep:99K epch:2464.88 loss:0.017 grdn:0.510 lr:5.0e-05 updt_s:0.327 data_s:0.024
INFO 2026-08-11 06:17:59 ot_train.py:423 step:172K smpl:88M ep:99K epch:2467.75 loss:0.017 grdn:0.515 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 06:19:08 ot_train.py:423 step:173K smpl:88M ep:99K epch:2470.61 loss:0.017 grdn:0.519 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 06:20:19 ot_train.py:423 step:173K smpl:88M ep:99K epch:2473.47 loss:0.018 grdn:0.531 lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-11 06:21:28 ot_train.py:423 step:173K smpl:89M ep:99K epch:2476.33 loss:0.017 grdn:0.524 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 06:22:38 ot_train.py:423 step:173K smpl:89M ep:99K epch:2479.20 loss:0.017 grdn:0.501 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 06:23:48 ot_train.py:423 step:173K smpl:89M ep:99K epch:2482.06 loss:0.017 grdn:0.474 lr:5.0e-05 updt_s:0.323 data_s:0.026
INFO 2026-08-11 06:24:57 ot_train.py:423 step:174K smpl:89M ep:99K epch:2484.92 loss:0.017 grdn:inf lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 06:26:06 ot_train.py:423 step:174K smpl:89M ep:100K epch:2487.79 loss:0.017 grdn:0.526 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 06:27:17 ot_train.py:423 step:174K smpl:89M ep:100K epch:2490.65 loss:0.017 grdn:0.529 lr:5.0e-05 updt_s:0.329 data_s:0.025
INFO 2026-08-11 06:27:17 ot_train.py:443 Checkpoint policy after step 174000
INFO 2026-08-11 06:28:27 ot_train.py:423 step:174K smpl:89M ep:100K epch:2493.51 loss:0.017 grdn:0.536 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 06:29:36 ot_train.py:423 step:174K smpl:89M ep:100K epch:2496.37 loss:0.017 grdn:0.506 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 06:30:47 ot_train.py:423 step:175K smpl:89M ep:100K epch:2499.24 loss:0.017 grdn:0.494 lr:5.0e-05 updt_s:0.328 data_s:0.025
INFO 2026-08-11 06:31:56 ot_train.py:423 step:175K smpl:89M ep:100K epch:2502.10 loss:0.017 grdn:0.543 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 06:33:07 ot_train.py:423 step:175K smpl:90M ep:100K epch:2504.96 loss:0.017 grdn:0.474 lr:5.0e-05 updt_s:0.330 data_s:0.024
INFO 2026-08-11 06:34:16 ot_train.py:423 step:175K smpl:90M ep:100K epch:2507.83 loss:0.017 grdn:0.460 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 06:35:25 ot_train.py:423 step:175K smpl:90M ep:100K epch:2510.69 loss:0.017 grdn:0.559 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 06:36:36 ot_train.py:423 step:176K smpl:90M ep:101K epch:2513.55 loss:0.017 grdn:inf lr:5.0e-05 updt_s:0.326 data_s:0.025
INFO 2026-08-11 06:37:45 ot_train.py:423 step:176K smpl:90M ep:101K epch:2516.41 loss:0.017 grdn:0.493 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 06:38:54 ot_train.py:423 step:176K smpl:90M ep:101K epch:2519.28 loss:0.017 grdn:0.514 lr:5.0e-05 updt_s:0.323 data_s:0.019
INFO 2026-08-11 06:38:54 ot_train.py:443 Checkpoint policy after step 176000
INFO 2026-08-11 06:40:05 ot_train.py:423 step:176K smpl:90M ep:101K epch:2522.14 loss:0.017 grdn:0.511 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-11 06:41:15 ot_train.py:423 step:176K smpl:90M ep:101K epch:2525.00 loss:0.017 grdn:0.517 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 06:42:24 ot_train.py:423 step:177K smpl:90M ep:101K epch:2527.86 loss:0.017 grdn:0.486 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 06:43:35 ot_train.py:423 step:177K smpl:91M ep:101K epch:2530.73 loss:0.017 grdn:0.527 lr:5.0e-05 updt_s:0.330 data_s:0.025
INFO 2026-08-11 06:44:44 ot_train.py:423 step:177K smpl:91M ep:101K epch:2533.59 loss:0.017 grdn:0.517 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 06:45:54 ot_train.py:423 step:177K smpl:91M ep:101K epch:2536.45 loss:0.017 grdn:0.534 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 06:47:05 ot_train.py:423 step:177K smpl:91M ep:102K epch:2539.32 loss:0.017 grdn:0.530 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-11 06:48:14 ot_train.py:423 step:178K smpl:91M ep:102K epch:2542.18 loss:0.017 grdn:0.519 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 06:49:24 ot_train.py:423 step:178K smpl:91M ep:102K epch:2545.04 loss:0.017 grdn:inf lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-11 06:50:34 ot_train.py:423 step:178K smpl:91M ep:102K epch:2547.90 loss:0.017 grdn:0.501 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 06:50:34 ot_train.py:443 Checkpoint policy after step 178000
INFO 2026-08-11 06:51:44 ot_train.py:423 step:178K smpl:91M ep:102K epch:2550.77 loss:0.017 grdn:0.518 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 06:52:54 ot_train.py:423 step:178K smpl:91M ep:102K epch:2553.63 loss:0.017 grdn:0.479 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-11 06:54:04 ot_train.py:423 step:179K smpl:91M ep:102K epch:2556.49 loss:0.017 grdn:0.466 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 06:55:13 ot_train.py:423 step:179K smpl:92M ep:102K epch:2559.36 loss:0.017 grdn:0.534 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 06:56:23 ot_train.py:423 step:179K smpl:92M ep:102K epch:2562.22 loss:0.017 grdn:0.529 lr:5.0e-05 updt_s:0.326 data_s:0.025
INFO 2026-08-11 06:57:33 ot_train.py:423 step:179K smpl:92M ep:103K epch:2565.08 loss:0.017 grdn:0.502 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 06:58:42 ot_train.py:423 step:179K smpl:92M ep:103K epch:2567.94 loss:0.017 grdn:0.487 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 06:59:53 ot_train.py:423 step:180K smpl:92M ep:103K epch:2570.81 loss:0.017 grdn:0.529 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-11 07:01:02 ot_train.py:423 step:180K smpl:92M ep:103K epch:2573.67 loss:0.017 grdn:inf lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 07:02:13 ot_train.py:423 step:180K smpl:92M ep:103K epch:2576.53 loss:0.017 grdn:0.519 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-11 07:02:13 ot_train.py:443 Checkpoint policy after step 180000
INFO 2026-08-11 07:03:23 ot_train.py:423 step:180K smpl:92M ep:103K epch:2579.40 loss:0.017 grdn:0.497 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 07:04:32 ot_train.py:423 step:180K smpl:92M ep:103K epch:2582.26 loss:0.017 grdn:0.518 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 07:05:42 ot_train.py:423 step:181K smpl:92M ep:103K epch:2585.12 loss:0.017 grdn:0.538 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-11 07:06:52 ot_train.py:423 step:181K smpl:93M ep:104K epch:2587.98 loss:0.017 grdn:0.522 lr:5.0e-05 updt_s:0.329 data_s:0.019
INFO 2026-08-11 07:08:01 ot_train.py:423 step:181K smpl:93M ep:104K epch:2590.85 loss:0.017 grdn:0.511 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 07:09:13 ot_train.py:423 step:181K smpl:93M ep:104K epch:2593.71 loss:0.017 grdn:0.514 lr:5.0e-05 updt_s:0.330 data_s:0.026
INFO 2026-08-11 07:10:22 ot_train.py:423 step:181K smpl:93M ep:104K epch:2596.57 loss:0.017 grdn:0.481 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 07:11:31 ot_train.py:423 step:182K smpl:93M ep:104K epch:2599.44 loss:0.017 grdn:0.497 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 07:12:42 ot_train.py:423 step:182K smpl:93M ep:104K epch:2602.30 loss:0.017 grdn:inf lr:5.0e-05 updt_s:0.330 data_s:0.025
INFO 2026-08-11 07:13:52 ot_train.py:423 step:182K smpl:93M ep:104K epch:2605.16 loss:0.017 grdn:0.484 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 07:13:52 ot_train.py:443 Checkpoint policy after step 182000
INFO 2026-08-11 07:15:01 ot_train.py:423 step:182K smpl:93M ep:104K epch:2608.02 loss:0.017 grdn:0.522 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 07:16:13 ot_train.py:423 step:182K smpl:93M ep:104K epch:2610.89 loss:0.017 grdn:0.548 lr:5.0e-05 updt_s:0.330 data_s:0.025
INFO 2026-08-11 07:17:22 ot_train.py:423 step:183K smpl:93M ep:105K epch:2613.75 loss:0.017 grdn:0.512 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 07:18:33 ot_train.py:423 step:183K smpl:94M ep:105K epch:2616.61 loss:0.017 grdn:0.522 lr:5.0e-05 updt_s:0.329 data_s:0.025
INFO 2026-08-11 07:19:42 ot_train.py:423 step:183K smpl:94M ep:105K epch:2619.47 loss:0.017 grdn:0.495 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 07:20:51 ot_train.py:423 step:183K smpl:94M ep:105K epch:2622.34 loss:0.017 grdn:0.525 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 07:22:01 ot_train.py:423 step:183K smpl:94M ep:105K epch:2625.20 loss:0.017 grdn:0.496 lr:5.0e-05 updt_s:0.325 data_s:0.026
INFO 2026-08-11 07:23:11 ot_train.py:423 step:184K smpl:94M ep:105K epch:2628.06 loss:0.017 grdn:0.523 lr:5.0e-05 updt_s:0.329 data_s:0.019
INFO 2026-08-11 07:24:20 ot_train.py:423 step:184K smpl:94M ep:105K epch:2630.93 loss:0.017 grdn:inf lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 07:25:32 ot_train.py:423 step:184K smpl:94M ep:105K epch:2633.79 loss:0.017 grdn:0.517 lr:5.0e-05 updt_s:0.330 data_s:0.027
INFO 2026-08-11 07:25:32 ot_train.py:443 Checkpoint policy after step 184000
INFO 2026-08-11 07:26:41 ot_train.py:423 step:184K smpl:94M ep:105K epch:2636.65 loss:0.017 grdn:0.529 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 07:27:50 ot_train.py:423 step:184K smpl:94M ep:106K epch:2639.51 loss:0.017 grdn:0.506 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 07:29:01 ot_train.py:423 step:185K smpl:95M ep:106K epch:2642.38 loss:0.017 grdn:0.535 lr:5.0e-05 updt_s:0.326 data_s:0.025
INFO 2026-08-11 07:30:10 ot_train.py:423 step:185K smpl:95M ep:106K epch:2645.24 loss:0.016 grdn:0.480 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 07:31:19 ot_train.py:423 step:185K smpl:95M ep:106K epch:2648.10 loss:0.017 grdn:0.501 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 07:32:29 ot_train.py:423 step:185K smpl:95M ep:106K epch:2650.97 loss:0.017 grdn:0.499 lr:5.0e-05 updt_s:0.324 data_s:0.026
INFO 2026-08-11 07:33:39 ot_train.py:423 step:185K smpl:95M ep:106K epch:2653.83 loss:0.017 grdn:0.510 lr:5.0e-05 updt_s:0.329 data_s:0.019
INFO 2026-08-11 07:34:50 ot_train.py:423 step:186K smpl:95M ep:106K epch:2656.69 loss:0.017 grdn:0.502 lr:5.0e-05 updt_s:0.324 data_s:0.027
INFO 2026-08-11 07:35:59 ot_train.py:423 step:186K smpl:95M ep:106K epch:2659.55 loss:0.017 grdn:0.541 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 07:37:08 ot_train.py:423 step:186K smpl:95M ep:106K epch:2662.42 loss:0.017 grdn:inf lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 07:37:08 ot_train.py:443 Checkpoint policy after step 186000
INFO 2026-08-11 07:38:19 ot_train.py:423 step:186K smpl:95M ep:107K epch:2665.28 loss:0.017 grdn:0.552 lr:5.0e-05 updt_s:0.325 data_s:0.025
INFO 2026-08-11 07:39:29 ot_train.py:423 step:186K smpl:95M ep:107K epch:2668.14 loss:0.017 grdn:0.499 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 07:40:38 ot_train.py:423 step:187K smpl:96M ep:107K epch:2671.01 loss:0.017 grdn:0.499 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 07:41:49 ot_train.py:423 step:187K smpl:96M ep:107K epch:2673.87 loss:0.017 grdn:0.531 lr:5.0e-05 updt_s:0.330 data_s:0.025
INFO 2026-08-11 07:42:58 ot_train.py:423 step:187K smpl:96M ep:107K epch:2676.73 loss:0.017 grdn:0.544 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 07:44:07 ot_train.py:423 step:187K smpl:96M ep:107K epch:2679.59 loss:0.017 grdn:0.515 lr:5.0e-05 updt_s:0.323 data_s:0.019
INFO 2026-08-11 07:45:18 ot_train.py:423 step:187K smpl:96M ep:107K epch:2682.46 loss:0.017 grdn:0.532 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-11 07:46:27 ot_train.py:423 step:188K smpl:96M ep:107K epch:2685.32 loss:0.017 grdn:0.526 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 07:47:36 ot_train.py:423 step:188K smpl:96M ep:108K epch:2688.18 loss:0.017 grdn:0.498 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 07:48:47 ot_train.py:423 step:188K smpl:96M ep:108K epch:2691.05 loss:0.016 grdn:0.508 lr:5.0e-05 updt_s:0.328 data_s:0.027
INFO 2026-08-11 07:48:47 ot_train.py:443 Checkpoint policy after step 188000
INFO 2026-08-11 07:49:57 ot_train.py:423 step:188K smpl:96M ep:108K epch:2693.91 loss:0.017 grdn:inf lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 07:51:08 ot_train.py:423 step:188K smpl:96M ep:108K epch:2696.77 loss:0.017 grdn:0.506 lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-11 07:52:18 ot_train.py:423 step:189K smpl:97M ep:108K epch:2699.63 loss:0.017 grdn:0.478 lr:5.0e-05 updt_s:0.329 data_s:0.019
INFO 2026-08-11 07:53:26 ot_train.py:423 step:189K smpl:97M ep:108K epch:2702.50 loss:0.017 grdn:0.539 lr:5.0e-05 updt_s:0.323 data_s:0.019
INFO 2026-08-11 07:54:38 ot_train.py:423 step:189K smpl:97M ep:108K epch:2705.36 loss:0.016 grdn:0.492 lr:5.0e-05 updt_s:0.330 data_s:0.026
INFO 2026-08-11 07:55:47 ot_train.py:423 step:189K smpl:97M ep:108K epch:2708.22 loss:0.017 grdn:0.519 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 07:56:56 ot_train.py:423 step:189K smpl:97M ep:108K epch:2711.09 loss:0.017 grdn:0.506 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 07:58:07 ot_train.py:423 step:190K smpl:97M ep:109K epch:2713.95 loss:0.017 grdn:0.493 lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-11 07:59:16 ot_train.py:423 step:190K smpl:97M ep:109K epch:2716.81 loss:0.017 grdn:0.509 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 08:00:25 ot_train.py:423 step:190K smpl:97M ep:109K epch:2719.67 loss:0.017 grdn:0.520 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 08:00:25 ot_train.py:443 Checkpoint policy after step 190000
INFO 2026-08-11 08:01:37 ot_train.py:423 step:190K smpl:97M ep:109K epch:2722.54 loss:0.017 grdn:inf lr:5.0e-05 updt_s:0.328 data_s:0.027
INFO 2026-08-11 08:02:46 ot_train.py:423 step:190K smpl:97M ep:109K epch:2725.40 loss:0.016 grdn:0.534 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 08:03:56 ot_train.py:423 step:191K smpl:98M ep:109K epch:2728.26 loss:0.017 grdn:0.522 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 08:05:07 ot_train.py:423 step:191K smpl:98M ep:109K epch:2731.12 loss:0.016 grdn:0.481 lr:5.0e-05 updt_s:0.330 data_s:0.026
INFO 2026-08-11 08:06:17 ot_train.py:423 step:191K smpl:98M ep:109K epch:2733.99 loss:0.017 grdn:0.527 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 08:07:27 ot_train.py:423 step:191K smpl:98M ep:109K epch:2736.85 loss:0.017 grdn:0.524 lr:5.0e-05 updt_s:0.327 data_s:0.025
INFO 2026-08-11 08:08:37 ot_train.py:423 step:191K smpl:98M ep:110K epch:2739.71 loss:0.016 grdn:0.496 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 08:09:46 ot_train.py:423 step:192K smpl:98M ep:110K epch:2742.58 loss:0.017 grdn:0.513 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 08:10:57 ot_train.py:423 step:192K smpl:98M ep:110K epch:2745.44 loss:0.017 grdn:0.512 lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-11 08:12:06 ot_train.py:423 step:192K smpl:98M ep:110K epch:2748.30 loss:0.016 grdn:0.500 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 08:12:06 ot_train.py:443 Checkpoint policy after step 192000
INFO 2026-08-11 08:13:16 ot_train.py:423 step:192K smpl:98M ep:110K epch:2751.16 loss:0.016 grdn:inf lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 08:14:27 ot_train.py:423 step:192K smpl:99M ep:110K epch:2754.03 loss:0.016 grdn:0.504 lr:5.0e-05 updt_s:0.331 data_s:0.025
INFO 2026-08-11 08:15:36 ot_train.py:423 step:193K smpl:99M ep:110K epch:2756.89 loss:0.017 grdn:0.497 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 08:16:46 ot_train.py:423 step:193K smpl:99M ep:110K epch:2759.75 loss:0.017 grdn:0.482 lr:5.0e-05 updt_s:0.326 data_s:0.020
INFO 2026-08-11 08:17:56 ot_train.py:423 step:193K smpl:99M ep:111K epch:2762.62 loss:0.016 grdn:0.504 lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-11 08:19:05 ot_train.py:423 step:193K smpl:99M ep:111K epch:2765.48 loss:0.017 grdn:0.498 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 08:20:15 ot_train.py:423 step:193K smpl:99M ep:111K epch:2768.34 loss:0.017 grdn:0.532 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 08:21:26 ot_train.py:423 step:194K smpl:99M ep:111K epch:2771.20 loss:0.017 grdn:0.524 lr:5.0e-05 updt_s:0.330 data_s:0.026
INFO 2026-08-11 08:22:35 ot_train.py:423 step:194K smpl:99M ep:111K epch:2774.07 loss:0.017 grdn:0.486 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 08:23:46 ot_train.py:423 step:194K smpl:99M ep:111K epch:2776.93 loss:0.017 grdn:0.532 lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-11 08:23:46 ot_train.py:443 Checkpoint policy after step 194000
INFO 2026-08-11 08:24:56 ot_train.py:423 step:194K smpl:99M ep:111K epch:2779.79 loss:0.017 grdn:inf lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 08:26:05 ot_train.py:423 step:194K smpl:100M ep:111K epch:2782.66 loss:0.016 grdn:0.527 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 08:27:15 ot_train.py:423 step:195K smpl:100M ep:111K epch:2785.52 loss:0.016 grdn:0.508 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-11 08:28:25 ot_train.py:423 step:195K smpl:100M ep:112K epch:2788.38 loss:0.017 grdn:0.521 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 08:29:34 ot_train.py:423 step:195K smpl:100M ep:112K epch:2791.24 loss:0.016 grdn:0.506 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 08:30:45 ot_train.py:423 step:195K smpl:100M ep:112K epch:2794.11 loss:0.016 grdn:0.509 lr:5.0e-05 updt_s:0.330 data_s:0.024
INFO 2026-08-11 08:31:54 ot_train.py:423 step:195K smpl:100M ep:112K epch:2796.97 loss:0.017 grdn:0.486 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 08:33:03 ot_train.py:423 step:196K smpl:100M ep:112K epch:2799.83 loss:0.017 grdn:0.540 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 08:34:14 ot_train.py:423 step:196K smpl:100M ep:112K epch:2802.70 loss:0.016 grdn:0.513 lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-11 08:35:24 ot_train.py:423 step:196K smpl:100M ep:112K epch:2805.56 loss:0.016 grdn:0.482 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 08:35:24 ot_train.py:443 Checkpoint policy after step 196000
INFO 2026-08-11 08:36:33 ot_train.py:423 step:196K smpl:100M ep:112K epch:2808.42 loss:0.018 grdn:0.594 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 08:37:44 ot_train.py:423 step:196K smpl:101M ep:112K epch:2811.28 loss:0.017 grdn:inf lr:5.0e-05 updt_s:0.329 data_s:0.026
INFO 2026-08-11 08:38:53 ot_train.py:423 step:197K smpl:101M ep:113K epch:2814.15 loss:0.017 grdn:0.504 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 08:40:04 ot_train.py:423 step:197K smpl:101M ep:113K epch:2817.01 loss:0.016 grdn:0.489 lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-11 08:41:13 ot_train.py:423 step:197K smpl:101M ep:113K epch:2819.87 loss:0.016 grdn:0.482 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 08:42:22 ot_train.py:423 step:197K smpl:101M ep:113K epch:2822.73 loss:0.016 grdn:0.499 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 08:43:33 ot_train.py:423 step:197K smpl:101M ep:113K epch:2825.60 loss:0.016 grdn:0.524 lr:5.0e-05 updt_s:0.329 data_s:0.025
INFO 2026-08-11 08:44:42 ot_train.py:423 step:198K smpl:101M ep:113K epch:2828.46 loss:0.016 grdn:0.522 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 08:45:52 ot_train.py:423 step:198K smpl:101M ep:113K epch:2831.32 loss:0.016 grdn:0.505 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 08:47:03 ot_train.py:423 step:198K smpl:101M ep:113K epch:2834.19 loss:0.016 grdn:0.468 lr:5.0e-05 updt_s:0.328 data_s:0.027
INFO 2026-08-11 08:47:03 ot_train.py:443 Checkpoint policy after step 198000
INFO 2026-08-11 08:48:13 ot_train.py:423 step:198K smpl:101M ep:113K epch:2837.05 loss:0.016 grdn:0.501 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 08:49:22 ot_train.py:423 step:198K smpl:102M ep:114K epch:2839.91 loss:0.017 grdn:0.504 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 08:50:33 ot_train.py:423 step:199K smpl:102M ep:114K epch:2842.77 loss:0.016 grdn:inf lr:5.0e-05 updt_s:0.329 data_s:0.026
INFO 2026-08-11 08:51:42 ot_train.py:423 step:199K smpl:102M ep:114K epch:2845.64 loss:0.016 grdn:0.530 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 08:52:51 ot_train.py:423 step:199K smpl:102M ep:114K epch:2848.50 loss:0.017 grdn:0.488 lr:5.0e-05 updt_s:0.323 data_s:0.019
INFO 2026-08-11 08:54:02 ot_train.py:423 step:199K smpl:102M ep:114K epch:2851.36 loss:0.016 grdn:0.526 lr:5.0e-05 updt_s:0.329 data_s:0.027
INFO 2026-08-11 08:55:11 ot_train.py:423 step:199K smpl:102M ep:114K epch:2854.23 loss:0.016 grdn:0.508 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 08:56:22 ot_train.py:423 step:200K smpl:102M ep:114K epch:2857.09 loss:0.016 grdn:0.509 lr:5.0e-05 updt_s:0.325 data_s:0.027
INFO 2026-08-11 08:57:31 ot_train.py:423 step:200K smpl:102M ep:114K epch:2859.95 loss:0.017 grdn:0.478 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 08:58:40 ot_train.py:423 step:200K smpl:102M ep:115K epch:2862.81 loss:0.016 grdn:0.508 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 08:58:40 ot_train.py:443 Checkpoint policy after step 200000
INFO 2026-08-11 08:59:51 ot_train.py:423 step:200K smpl:103M ep:115K epch:2865.68 loss:0.016 grdn:0.522 lr:5.0e-05 updt_s:0.326 data_s:0.025
INFO 2026-08-11 09:01:01 ot_train.py:423 step:200K smpl:103M ep:115K epch:2868.54 loss:0.016 grdn:0.489 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 09:02:10 ot_train.py:423 step:201K smpl:103M ep:115K epch:2871.40 loss:0.017 grdn:inf lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 09:03:20 ot_train.py:423 step:201K smpl:103M ep:115K epch:2874.27 loss:0.016 grdn:0.523 lr:5.0e-05 updt_s:0.327 data_s:0.025
INFO 2026-08-11 09:04:30 ot_train.py:423 step:201K smpl:103M ep:115K epch:2877.13 loss:0.016 grdn:0.496 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 09:05:39 ot_train.py:423 step:201K smpl:103M ep:115K epch:2879.99 loss:0.016 grdn:0.495 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 09:06:50 ot_train.py:423 step:201K smpl:103M ep:115K epch:2882.85 loss:0.016 grdn:0.526 lr:5.0e-05 updt_s:0.329 data_s:0.026
INFO 2026-08-11 09:07:59 ot_train.py:423 step:202K smpl:103M ep:115K epch:2885.72 loss:0.016 grdn:0.477 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 09:09:10 ot_train.py:423 step:202K smpl:103M ep:116K epch:2888.58 loss:0.016 grdn:0.499 lr:5.0e-05 updt_s:0.328 data_s:0.025
INFO 2026-08-11 09:10:19 ot_train.py:423 step:202K smpl:103M ep:116K epch:2891.44 loss:0.017 grdn:0.528 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 09:10:19 ot_train.py:443 Checkpoint policy after step 202000
INFO 2026-08-11 09:11:29 ot_train.py:423 step:202K smpl:104M ep:116K epch:2894.31 loss:0.016 grdn:0.494 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 09:12:40 ot_train.py:423 step:202K smpl:104M ep:116K epch:2897.17 loss:0.017 grdn:0.521 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-11 09:13:49 ot_train.py:423 step:203K smpl:104M ep:116K epch:2900.03 loss:0.016 grdn:0.507 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 09:14:58 ot_train.py:423 step:203K smpl:104M ep:116K epch:2902.89 loss:0.017 grdn:inf lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 09:16:09 ot_train.py:423 step:203K smpl:104M ep:116K epch:2905.76 loss:0.016 grdn:0.495 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-11 09:17:18 ot_train.py:423 step:203K smpl:104M ep:116K epch:2908.62 loss:0.016 grdn:0.540 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 09:18:27 ot_train.py:423 step:203K smpl:104M ep:116K epch:2911.48 loss:0.016 grdn:0.514 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 09:19:38 ot_train.py:423 step:204K smpl:104M ep:117K epch:2914.34 loss:0.016 grdn:0.514 lr:5.0e-05 updt_s:0.329 data_s:0.026
INFO 2026-08-11 09:20:48 ot_train.py:423 step:204K smpl:104M ep:117K epch:2917.21 loss:0.016 grdn:0.471 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 09:21:57 ot_train.py:423 step:204K smpl:104M ep:117K epch:2920.07 loss:0.016 grdn:0.495 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 09:21:57 ot_train.py:443 Checkpoint policy after step 204000
INFO 2026-08-11 09:23:08 ot_train.py:423 step:204K smpl:105M ep:117K epch:2922.93 loss:0.016 grdn:0.549 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-11 09:24:18 ot_train.py:423 step:204K smpl:105M ep:117K epch:2925.80 loss:0.017 grdn:0.583 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 09:25:28 ot_train.py:423 step:205K smpl:105M ep:117K epch:2928.66 loss:0.016 grdn:0.492 lr:5.0e-05 updt_s:0.327 data_s:0.024
INFO 2026-08-11 09:26:38 ot_train.py:423 step:205K smpl:105M ep:117K epch:2931.52 loss:0.016 grdn:0.553 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 09:27:47 ot_train.py:423 step:205K smpl:105M ep:117K epch:2934.38 loss:0.016 grdn:inf lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 09:28:58 ot_train.py:423 step:205K smpl:105M ep:117K epch:2937.25 loss:0.016 grdn:0.511 lr:5.0e-05 updt_s:0.330 data_s:0.026
INFO 2026-08-11 09:30:07 ot_train.py:423 step:205K smpl:105M ep:118K epch:2940.11 loss:0.016 grdn:0.485 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 09:31:16 ot_train.py:423 step:206K smpl:105M ep:118K epch:2942.97 loss:0.016 grdn:0.516 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 09:32:27 ot_train.py:423 step:206K smpl:105M ep:118K epch:2945.84 loss:0.016 grdn:0.496 lr:5.0e-05 updt_s:0.328 data_s:0.025
INFO 2026-08-11 09:33:37 ot_train.py:423 step:206K smpl:105M ep:118K epch:2948.70 loss:0.016 grdn:0.499 lr:5.0e-05 updt_s:0.329 data_s:0.019
INFO 2026-08-11 09:33:37 ot_train.py:443 Checkpoint policy after step 206000
INFO 2026-08-11 09:34:47 ot_train.py:423 step:206K smpl:106M ep:118K epch:2951.56 loss:0.016 grdn:0.513 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 09:35:57 ot_train.py:423 step:206K smpl:106M ep:118K epch:2954.42 loss:0.016 grdn:0.534 lr:5.0e-05 updt_s:0.324 data_s:0.025
INFO 2026-08-11 09:37:06 ot_train.py:423 step:207K smpl:106M ep:118K epch:2957.29 loss:0.016 grdn:0.480 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 09:38:16 ot_train.py:423 step:207K smpl:106M ep:118K epch:2960.15 loss:0.016 grdn:0.532 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 09:39:26 ot_train.py:423 step:207K smpl:106M ep:119K epch:2963.01 loss:0.016 grdn:0.503 lr:5.0e-05 updt_s:0.327 data_s:0.025
INFO 2026-08-11 09:40:36 ot_train.py:423 step:207K smpl:106M ep:119K epch:2965.88 loss:0.016 grdn:inf lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 09:41:46 ot_train.py:423 step:207K smpl:106M ep:119K epch:2968.74 loss:0.016 grdn:0.523 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-11 09:42:55 ot_train.py:423 step:208K smpl:106M ep:119K epch:2971.60 loss:0.016 grdn:0.478 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 09:44:05 ot_train.py:423 step:208K smpl:106M ep:119K epch:2974.46 loss:0.016 grdn:0.496 lr:5.0e-05 updt_s:0.326 data_s:0.020
INFO 2026-08-11 09:45:15 ot_train.py:423 step:208K smpl:106M ep:119K epch:2977.33 loss:0.016 grdn:0.580 lr:5.0e-05 updt_s:0.328 data_s:0.025
INFO 2026-08-11 09:45:15 ot_train.py:443 Checkpoint policy after step 208000
INFO 2026-08-11 09:46:25 ot_train.py:423 step:208K smpl:107M ep:119K epch:2980.19 loss:0.016 grdn:0.497 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 09:47:34 ot_train.py:423 step:208K smpl:107M ep:119K epch:2983.05 loss:0.016 grdn:0.521 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 09:48:45 ot_train.py:423 step:209K smpl:107M ep:119K epch:2985.92 loss:0.016 grdn:0.527 lr:5.0e-05 updt_s:0.329 data_s:0.026
INFO 2026-08-11 09:49:54 ot_train.py:423 step:209K smpl:107M ep:120K epch:2988.78 loss:0.016 grdn:0.489 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 09:51:04 ot_train.py:423 step:209K smpl:107M ep:120K epch:2991.64 loss:0.016 grdn:0.490 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 09:52:15 ot_train.py:423 step:209K smpl:107M ep:120K epch:2994.50 loss:0.016 grdn:inf lr:5.0e-05 updt_s:0.328 data_s:0.027
INFO 2026-08-11 09:53:24 ot_train.py:423 step:209K smpl:107M ep:120K epch:2997.37 loss:0.016 grdn:0.514 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 09:54:33 ot_train.py:423 step:210K smpl:107M ep:120K epch:3000.23 loss:0.016 grdn:0.502 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 09:55:44 ot_train.py:423 step:210K smpl:107M ep:120K epch:3003.09 loss:0.016 grdn:0.491 lr:5.0e-05 updt_s:0.329 data_s:0.025
INFO 2026-08-11 09:56:54 ot_train.py:423 step:210K smpl:108M ep:120K epch:3005.95 loss:0.016 grdn:0.483 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 09:56:54 ot_train.py:443 Checkpoint policy after step 210000
INFO 2026-08-11 09:58:05 ot_train.py:423 step:210K smpl:108M ep:120K epch:3008.82 loss:0.016 grdn:0.511 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-11 09:59:14 ot_train.py:423 step:210K smpl:108M ep:120K epch:3011.68 loss:0.016 grdn:0.485 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 10:00:23 ot_train.py:423 step:211K smpl:108M ep:121K epch:3014.54 loss:0.016 grdn:0.533 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 10:01:34 ot_train.py:423 step:211K smpl:108M ep:121K epch:3017.41 loss:0.017 grdn:0.549 lr:5.0e-05 updt_s:0.329 data_s:0.026
INFO 2026-08-11 10:02:44 ot_train.py:423 step:211K smpl:108M ep:121K epch:3020.27 loss:0.016 grdn:0.505 lr:5.0e-05 updt_s:0.325 data_s:0.020
INFO 2026-08-11 10:03:53 ot_train.py:423 step:211K smpl:108M ep:121K epch:3023.13 loss:0.017 grdn:inf lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 10:05:03 ot_train.py:423 step:211K smpl:108M ep:121K epch:3025.99 loss:0.016 grdn:0.509 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-11 10:06:13 ot_train.py:423 step:212K smpl:108M ep:121K epch:3028.86 loss:0.016 grdn:0.498 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 10:07:22 ot_train.py:423 step:212K smpl:108M ep:121K epch:3031.72 loss:0.016 grdn:0.478 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 10:08:33 ot_train.py:423 step:212K smpl:109M ep:121K epch:3034.58 loss:0.016 grdn:0.520 lr:5.0e-05 updt_s:0.326 data_s:0.027
INFO 2026-08-11 10:08:33 ot_train.py:443 Checkpoint policy after step 212000
INFO 2026-08-11 10:09:42 ot_train.py:423 step:212K smpl:109M ep:121K epch:3037.45 loss:0.016 grdn:0.499 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 10:10:51 ot_train.py:423 step:212K smpl:109M ep:122K epch:3040.31 loss:0.016 grdn:0.517 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 10:12:02 ot_train.py:423 step:213K smpl:109M ep:122K epch:3043.17 loss:0.015 grdn:0.496 lr:5.0e-05 updt_s:0.329 data_s:0.026
INFO 2026-08-11 10:13:12 ot_train.py:423 step:213K smpl:109M ep:122K epch:3046.03 loss:0.016 grdn:0.528 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 10:14:22 ot_train.py:423 step:213K smpl:109M ep:122K epch:3048.90 loss:0.016 grdn:0.486 lr:5.0e-05 updt_s:0.324 data_s:0.027
INFO 2026-08-11 10:15:32 ot_train.py:423 step:213K smpl:109M ep:122K epch:3051.76 loss:0.016 grdn:inf lr:5.0e-05 updt_s:0.329 data_s:0.019
INFO 2026-08-11 10:16:41 ot_train.py:423 step:213K smpl:109M ep:122K epch:3054.62 loss:0.016 grdn:0.505 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 10:17:52 ot_train.py:423 step:214K smpl:109M ep:122K epch:3057.49 loss:0.016 grdn:0.511 lr:5.0e-05 updt_s:0.328 data_s:0.026
INFO 2026-08-11 10:19:01 ot_train.py:423 step:214K smpl:109M ep:122K epch:3060.35 loss:0.016 grdn:0.531 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 10:20:11 ot_train.py:423 step:214K smpl:110M ep:123K epch:3063.21 loss:0.016 grdn:0.544 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 10:20:11 ot_train.py:443 Checkpoint policy after step 214000
INFO 2026-08-11 10:21:22 ot_train.py:423 step:214K smpl:110M ep:123K epch:3066.07 loss:0.016 grdn:0.496 lr:5.0e-05 updt_s:0.327 data_s:0.026
INFO 2026-08-11 10:22:31 ot_train.py:423 step:214K smpl:110M ep:123K epch:3068.94 loss:0.016 grdn:0.511 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 10:23:40 ot_train.py:423 step:215K smpl:110M ep:123K epch:3071.80 loss:0.016 grdn:0.531 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 10:24:51 ot_train.py:423 step:215K smpl:110M ep:123K epch:3074.66 loss:0.016 grdn:0.532 lr:5.0e-05 updt_s:0.327 data_s:0.025
INFO 2026-08-11 10:26:00 ot_train.py:423 step:215K smpl:110M ep:123K epch:3077.53 loss:0.016 grdn:0.489 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 10:27:09 ot_train.py:423 step:215K smpl:110M ep:123K epch:3080.39 loss:0.016 grdn:0.513 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 10:28:20 ot_train.py:423 step:215K smpl:110M ep:123K epch:3083.25 loss:0.016 grdn:0.502 lr:5.0e-05 updt_s:0.326 data_s:0.026
INFO 2026-08-11 10:29:29 ot_train.py:423 step:216K smpl:110M ep:123K epch:3086.11 loss:0.016 grdn:inf lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 10:30:40 ot_train.py:423 step:216K smpl:110M ep:124K epch:3088.98 loss:0.016 grdn:0.499 lr:5.0e-05 updt_s:0.329 data_s:0.026
INFO 2026-08-11 10:31:49 ot_train.py:423 step:216K smpl:111M ep:124K epch:3091.84 loss:0.016 grdn:0.500 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 10:31:49 ot_train.py:443 Checkpoint policy after step 216000
INFO 2026-08-11 10:32:59 ot_train.py:423 step:216K smpl:111M ep:124K epch:3094.70 loss:0.016 grdn:0.500 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 10:34:09 ot_train.py:423 step:216K smpl:111M ep:124K epch:3097.56 loss:0.016 grdn:0.514 lr:5.0e-05 updt_s:0.327 data_s:0.024
INFO 2026-08-11 10:35:19 ot_train.py:423 step:217K smpl:111M ep:124K epch:3100.43 loss:0.016 grdn:0.517 lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 10:36:28 ot_train.py:423 step:217K smpl:111M ep:124K epch:3103.29 loss:0.016 grdn:0.499 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 10:37:39 ot_train.py:423 step:217K smpl:111M ep:124K epch:3106.15 loss:0.016 grdn:0.509 lr:5.0e-05 updt_s:0.330 data_s:0.025
INFO 2026-08-11 10:38:48 ot_train.py:423 step:217K smpl:111M ep:124K epch:3109.02 loss:0.016 grdn:0.501 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 10:39:57 ot_train.py:423 step:217K smpl:111M ep:124K epch:3111.88 loss:0.016 grdn:0.484 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 10:41:08 ot_train.py:423 step:218K smpl:111M ep:125K epch:3114.74 loss:0.016 grdn:0.505 lr:5.0e-05 updt_s:0.327 data_s:0.025
INFO 2026-08-11 10:42:17 ot_train.py:423 step:218K smpl:112M ep:125K epch:3117.60 loss:0.016 grdn:inf lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 10:43:26 ot_train.py:423 step:218K smpl:112M ep:125K epch:3120.47 loss:0.016 grdn:0.514 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 10:43:26 ot_train.py:443 Checkpoint policy after step 218000
INFO 2026-08-11 10:44:37 ot_train.py:423 step:218K smpl:112M ep:125K epch:3123.33 loss:0.016 grdn:0.527 lr:5.0e-05 updt_s:0.326 data_s:0.025
INFO 2026-08-11 10:45:47 ot_train.py:423 step:218K smpl:112M ep:125K epch:3126.19 loss:0.016 grdn:0.477 lr:5.0e-05 updt_s:0.329 data_s:0.019
INFO 2026-08-11 10:46:57 ot_train.py:423 step:219K smpl:112M ep:125K epch:3129.06 loss:0.016 grdn:0.538 lr:5.0e-05 updt_s:0.326 data_s:0.025
INFO 2026-08-11 10:48:06 ot_train.py:423 step:219K smpl:112M ep:125K epch:3131.92 loss:0.016 grdn:0.521 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 10:49:15 ot_train.py:423 step:219K smpl:112M ep:125K epch:3134.78 loss:0.016 grdn:0.515 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 10:50:27 ot_train.py:423 step:219K smpl:112M ep:126K epch:3137.64 loss:0.016 grdn:0.487 lr:5.0e-05 updt_s:0.329 data_s:0.026
INFO 2026-08-11 10:51:36 ot_train.py:423 step:219K smpl:112M ep:126K epch:3140.51 loss:0.016 grdn:0.508 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 10:52:45 ot_train.py:423 step:220K smpl:112M ep:126K epch:3143.37 loss:0.016 grdn:0.514 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 10:53:56 ot_train.py:423 step:220K smpl:113M ep:126K epch:3146.23 loss:0.016 grdn:0.514 lr:5.0e-05 updt_s:0.329 data_s:0.025
INFO 2026-08-11 10:55:05 ot_train.py:423 step:220K smpl:113M ep:126K epch:3149.10 loss:0.016 grdn:inf lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 10:55:05 ot_train.py:443 Checkpoint policy after step 220000
INFO 2026-08-11 10:56:15 ot_train.py:423 step:220K smpl:113M ep:126K epch:3151.96 loss:0.016 grdn:0.501 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 10:57:25 ot_train.py:423 step:220K smpl:113M ep:126K epch:3154.82 loss:0.016 grdn:0.525 lr:5.0e-05 updt_s:0.324 data_s:0.026
INFO 2026-08-11 10:58:34 ot_train.py:423 step:221K smpl:113M ep:126K epch:3157.68 loss:0.016 grdn:0.519 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 10:59:43 ot_train.py:423 step:221K smpl:113M ep:126K epch:3160.55 loss:0.016 grdn:0.487 lr:5.0e-05 updt_s:0.324 data_s:0.019
INFO 2026-08-11 11:00:54 ot_train.py:423 step:221K smpl:113M ep:127K epch:3163.41 loss:0.016 grdn:0.533 lr:5.0e-05 updt_s:0.326 data_s:0.027
INFO 2026-08-11 11:02:04 ot_train.py:423 step:221K smpl:113M ep:127K epch:3166.27 loss:0.016 grdn:0.524 lr:5.0e-05 updt_s:0.329 data_s:0.019
INFO 2026-08-11 11:03:14 ot_train.py:423 step:221K smpl:113M ep:127K epch:3169.14 loss:0.016 grdn:0.533 lr:5.0e-05 updt_s:0.325 data_s:0.026
INFO 2026-08-11 11:04:24 ot_train.py:423 step:222K smpl:113M ep:127K epch:3172.00 loss:0.016 grdn:0.496 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 11:05:33 ot_train.py:423 step:222K smpl:114M ep:127K epch:3174.86 loss:0.016 grdn:0.532 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 11:06:44 ot_train.py:423 step:222K smpl:114M ep:127K epch:3177.72 loss:0.016 grdn:0.500 lr:5.0e-05 updt_s:0.328 data_s:0.025
INFO 2026-08-11 11:06:44 ot_train.py:443 Checkpoint policy after step 222000
INFO 2026-08-11 11:07:54 ot_train.py:423 step:222K smpl:114M ep:127K epch:3180.59 loss:0.016 grdn:inf lr:5.0e-05 updt_s:0.328 data_s:0.019
INFO 2026-08-11 11:09:03 ot_train.py:423 step:222K smpl:114M ep:127K epch:3183.45 loss:0.016 grdn:0.550 lr:5.0e-05 updt_s:0.327 data_s:0.019
INFO 2026-08-11 11:10:14 ot_train.py:423 step:223K smpl:114M ep:127K epch:3186.31 loss:0.016 grdn:0.485 lr:5.0e-05 updt_s:0.326 data_s:0.027
INFO 2026-08-11 11:11:23 ot_train.py:423 step:223K smpl:114M ep:128K epch:3189.17 loss:0.016 grdn:0.500 lr:5.0e-05 updt_s:0.325 data_s:0.019
INFO 2026-08-11 11:12:32 ot_train.py:423 step:223K smpl:114M ep:128K epch:3192.04 loss:0.016 grdn:0.525 lr:5.0e-05 updt_s:0.326 data_s:0.019
INFO 2026-08-11 11:13:43 ot_train.py:423 step:223K smpl:114M ep:128K epch:3194.90 loss:0.016 grdn:0.531 lr:5.0e-05 updt_s:0.330 data_s:0.025"""

# Parse the log data
def parse_value(val_str):
    """Parse values like '1K', '1M', 'nan', 'inf' to numeric values"""
    val_str = val_str.strip()
    if val_str == 'nan':
        return np.nan
    if val_str == 'inf':
        return np.nan  # Treat inf as nan for plotting

    if 'K' in val_str:
        return float(val_str.replace('K', '')) * 1000
    elif 'M' in val_str:
        return float(val_str.replace('M', '')) * 1000000
    else:
        try:
            return float(val_str)
        except:
            return np.nan

# Initialize lists to store parsed data
steps = []
samples = []
episodes = []
epochs = []
losses = []
gradients = []
learning_rates = []
update_times = []
data_times = []

# Parse each line
for line in log_data.strip().split('\n'):
    if 'step:' not in line:
        continue

    # Extract values using regex
    step_match = re.search(r'step:(\S+)', line)
    smpl_match = re.search(r'smpl:(\S+)', line)
    ep_match = re.search(r'ep:(\S+)', line)
    epch_match = re.search(r'epch:(\S+)', line)
    loss_match = re.search(r'loss:(\S+)', line)
    grdn_match = re.search(r'grdn:(\S+)', line)
    lr_match = re.search(r'lr:(\S+)', line)
    updt_match = re.search(r'updt_s:(\S+)', line)
    data_match = re.search(r'data_s:(\S+)', line)

    if step_match:
        steps.append(parse_value(step_match.group(1)))
        samples.append(parse_value(smpl_match.group(1)) if smpl_match else np.nan)
        episodes.append(parse_value(ep_match.group(1)) if ep_match else np.nan)
        epochs.append(parse_value(epch_match.group(1)) if epch_match else np.nan)
        losses.append(parse_value(loss_match.group(1)) if loss_match else np.nan)
        gradients.append(parse_value(grdn_match.group(1)) if grdn_match else np.nan)
        learning_rates.append(parse_value(lr_match.group(1)) if lr_match else np.nan)
        update_times.append(parse_value(updt_match.group(1)) if updt_match else np.nan)
        data_times.append(parse_value(data_match.group(1)) if data_match else np.nan)

# Create visualizations
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('Training Metrics Visualization (Steps 38K-50K)', fontsize=16, fontweight='bold')

# Plot 1: Loss over steps
axes[0, 0].plot(steps, losses, 'b-', linewidth=2, marker='o', markersize=3)
axes[0, 0].set_xlabel('Training Steps')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Gradient norm over steps (filter out nan/inf)
valid_grads = [(s, g) for s, g in zip(steps, gradients) if not np.isnan(g)]
if valid_grads:
    grad_steps, grad_values = zip(*valid_grads)
    axes[0, 1].plot(grad_steps, grad_values, 'r-', linewidth=2, marker='o', markersize=3)
axes[0, 1].set_xlabel('Training Steps')
axes[0, 1].set_ylabel('Gradient Norm')
axes[0, 1].set_title('Gradient Norm (inf values excluded)')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Epochs over steps
axes[1, 0].plot(steps, epochs, 'g-', linewidth=2, marker='o', markersize=3)
axes[1, 0].set_xlabel('Training Steps')
axes[1, 0].set_ylabel('Epochs')
axes[1, 0].set_title('Training Progress (Epochs)')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Update time and data time
axes[1, 1].plot(steps, update_times, 'purple', linewidth=2, marker='o', markersize=3, label='Update Time')
axes[1, 1].plot(steps, data_times, 'orange', linewidth=2, marker='s', markersize=3, label='Data Time')
axes[1, 1].set_xlabel('Training Steps')
axes[1, 1].set_ylabel('Time (seconds)')
axes[1, 1].set_title('Update and Data Loading Time')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Plot 5: Samples processed over steps
axes[2, 0].plot(steps, samples, 'cyan', linewidth=2, marker='o', markersize=3)
axes[2, 0].set_xlabel('Training Steps')
axes[2, 0].set_ylabel('Samples Processed')
axes[2, 0].set_title('Total Samples Processed')
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].ticklabel_format(style='plain', axis='y')

# Plot 6: Episodes over steps
axes[2, 1].plot(steps, episodes, 'magenta', linewidth=2, marker='o', markersize=3)
axes[2, 1].set_xlabel('Training Steps')
axes[2, 1].set_ylabel('Episodes')
axes[2, 1].set_title('Episodes Processed')
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ksa/lerobot/current_training_visualization.png', dpi=300, bbox_inches='tight')
print(f"Visualization saved to: /home/ksa/lerobot/current_training_visualization.png")
print(f"\nTraining Summary:")
print(f"  Steps: {int(steps[0])} -> {int(steps[-1])}")
print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
print(f"  Epochs: {epochs[0]:.2f} -> {epochs[-1]:.2f}")
print(f"  Samples: {int(samples[0]/1e6)}M -> {int(samples[-1]/1e6)}M")
print(f"  Episodes: {int(episodes[0]/1000)}K -> {int(episodes[-1]/1000)}K")
plt.show()
