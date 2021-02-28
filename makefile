
weight_histogram:
	python3 histogram.py
visualize_raw:
	python3 draw_simple_graph.py > graph.dot
	sfdp -x -Goverlap=scale -Tpng graph.dot > data.png

clean:
	-@rm *.dot *.png
