#!/bin/sh

/home/usr1/15D54036/uni/array16/device

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


