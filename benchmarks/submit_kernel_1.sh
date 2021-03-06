#!/bin/sh

/home/usr1/15D54036/uni/array16/device

echo "COLUMN NO REORDER"
/home/usr1/15D54036/uni/array16/bench_column
/home/usr1/15D54036/uni/array16/bench_column
/home/usr1/15D54036/uni/array16/bench_column
/home/usr1/15D54036/uni/array16/bench_column
/home/usr1/15D54036/uni/array16/bench_column

echo "COLUMN WITH REORDER"
/home/usr1/15D54036/uni/array16/bench_column_reorder
/home/usr1/15D54036/uni/array16/bench_column_reorder
/home/usr1/15D54036/uni/array16/bench_column_reorder
/home/usr1/15D54036/uni/array16/bench_column_reorder
/home/usr1/15D54036/uni/array16/bench_column_reorder

echo "ROW NO REORDER"
/home/usr1/15D54036/uni/array16/bench_row
/home/usr1/15D54036/uni/array16/bench_row
/home/usr1/15D54036/uni/array16/bench_row
/home/usr1/15D54036/uni/array16/bench_row
/home/usr1/15D54036/uni/array16/bench_row

echo "ROW WITH REORDER"
/home/usr1/15D54036/uni/array16/bench_row_reorder
/home/usr1/15D54036/uni/array16/bench_row_reorder
/home/usr1/15D54036/uni/array16/bench_row_reorder
/home/usr1/15D54036/uni/array16/bench_row_reorder
/home/usr1/15D54036/uni/array16/bench_row_reorder

echo "ROW NO NO CACHE REORDER"
/home/usr1/15D54036/uni/array16/bench_row_nocache
/home/usr1/15D54036/uni/array16/bench_row_nocache
/home/usr1/15D54036/uni/array16/bench_row_nocache
/home/usr1/15D54036/uni/array16/bench_row_nocache
/home/usr1/15D54036/uni/array16/bench_row_nocache

echo "ROW WITH NO CACHE REORDER"
/home/usr1/15D54036/uni/array16/bench_row_reorder_nocache
/home/usr1/15D54036/uni/array16/bench_row_reorder_nocache
/home/usr1/15D54036/uni/array16/bench_row_reorder_nocache
/home/usr1/15D54036/uni/array16/bench_row_reorder_nocache
/home/usr1/15D54036/uni/array16/bench_row_reorder_nocache
