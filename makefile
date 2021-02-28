generate_graphs:
	python3 draw_simple_graph.py > graph.dot
	sfdp -x -Goverlap=scale -Tpng graph.dot > data.png
