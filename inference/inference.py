import cv2
import argparse
import numpy as np
from pathlib import Path

import util


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default='config.yaml')
	parser.add_argument('--type-inference', type=str)
	parser.add_argument('--input-dir')
	parser.add_argument('--pattern', type=str)
	parser.add_argument('--video-output', type=str)
	parser.add_argument('--show', action='store_true')
	parser.add_argument('--output-dir', default='output/')	
	args = parser.parse_args()

	output_dir = Path(args.output_dir)
	if not output_dir.exists():
	    output_dir.mkdir(parents=True)
	config = util.load_yaml(args.config)
	predictor = util.create_instance(config)

	if args.type_inference == 'webcam' or args.type_inference == 'video':
		if args.type_inference == 'webcam':
			cap = cv2.VideoCapture(int(args.input_dir))
		elif args.type_inference == 'video':
			cap = cv2.VideoCapture(args.input_dir)

		success, frame = cap.read()
		video_width, video_height = frame.shape[:2]

		fourcc = cv2.VideoWriter_fourcc(*"XVID")
		out = cv2.VideoWriter(args.video_output, fourcc, 10.0, (video_height, video_width))

		cap.set(3, video_width)
		cap.set(4, video_height)
		

		while True:
			success, image = cap.read()

			if success == True:
				prediction = predictor([image])[0]

				if prediction['labels'] is not None:
				    boxes = prediction['boxes'].cpu().numpy().astype(np.int32)
				    labels = prediction['labels'].cpu().numpy()
				    scores = prediction['scores'].cpu().numpy()
				    class_names = prediction['names']


				    font_scale = max(video_height, video_width) / 1200
				    box_thickness = max(video_height, video_width) // 500
				    text_thickness = max(video_height, video_width) // 500
				    for box, score, class_name in zip(boxes, scores, class_names):

				        color = (
				            np.random.randint(200, 255),
				            np.random.randint(50, 200),
				            np.random.randint(0, 150)
				        )

				        cv2.rectangle(
				            img=image,
				            pt1=tuple(box[:2]),
				            pt2=tuple(box[2:]),
				            color=color,
				            thickness=box_thickness
				        )

				        w_text, h_text = cv2.getTextSize(
				            f'{class_name}: {score.item() * 100: .2f}%',
				            cv2.FONT_HERSHEY_PLAIN,
				            font_scale,
				            text_thickness
				        )[0]

				        cv2.rectangle(
				            img=image,
				            pt1=(box[0], box[1] - int(1.5 * h_text)),
				            pt2=(box[0] + int(1.1 * w_text), box[1]),
				            color=color,
				            thickness=-1
				        )

				        cv2.putText(
				            img=image,
				            text=f'{class_name}: {score.item() * 100: .2f}%',
				            org=(box[0], box[1] - int(0.5 * h_text)),
				            fontFace=cv2.FONT_HERSHEY_PLAIN,
				            fontScale=font_scale,
				            color=(255, 255, 255),
				            thickness=text_thickness,
				            lineType=cv2.LINE_AA)
					        
				out.write(image)

				cv2.imshow("Output", image)	

				if cv2.waitKey(30) == ord('q'):
					break
			else:
				break

		cap.release()
		out.release()
		cv2.destroyAllWindows()

	elif args.type_inference == 'image':
		image_paths = list(Path(args.input_dir).glob(args.pattern)) if args.pattern else [Path(args.input_dir)]

		for i, image_path in enumerate(image_paths, 1):
		    print('**' * 30)
		    print(f'{i} / {len(image_paths)} - {image_path.name}')

		    image = cv2.imread(str(image_path))
		    image_height, image_width = image.shape[:2]
		    prediction = predictor([image])[0]

		    if prediction['labels'] is not None:
		        font_scale = max(image_height, image_width) / 1200
		        box_thickness = max(image_height, image_width) // 500
		        text_thickness = max(image_height, image_width) // 500
		        boxes = prediction['boxes'].cpu().numpy().astype(np.int32)
		        labels = prediction['labels'].cpu().numpy()
		        scores = prediction['scores'].cpu().numpy()
		        class_names = prediction['names']
		        for box, score, class_name in zip(boxes, scores, class_names):
			        color = (
			            np.random.randint(200, 255),
			            np.random.randint(50, 200),
			            np.random.randint(0, 150)
			        )

			        cv2.rectangle(
			            img=image,
			            pt1=tuple(box[:2]),
			            pt2=tuple(box[2:]),
			            color=color,
			            thickness=box_thickness
			        )

			        w_text, h_text = cv2.getTextSize(
			            f'{class_name}: {score.item() * 100: .2f}%',
			            cv2.FONT_HERSHEY_PLAIN,
			            font_scale,
			            text_thickness
			        )[0]

			        cv2.rectangle(
			            img=image,
			            pt1=(box[0], box[1] - int(1.5 * h_text)),
			            pt2=(box[0] + int(1.1 * w_text), box[1]),
			            color=color,
			            thickness=-1
			        )

			        cv2.putText(
			            img=image,
			            text=f'{class_name}: {score.item() * 100: .2f}%',
			            org=(box[0], box[1] - int(0.5 * h_text)),
			            fontFace=cv2.FONT_HERSHEY_PLAIN,
			            fontScale=font_scale,
			            color=(255, 255, 255),
			            thickness=text_thickness,
			            lineType=cv2.LINE_AA)
		        cv2.imshow(image_path.name, image)

		        if cv2.waitKey(30) == ord('q'):
		        	break
		        # if args.show:
		cv2.imwrite(str(output_dir.joinpath(image_path.name)), image)
		cv2.waitKey()
		cv2.destroyAllWindows()
	
	else:
		print('not have this type to inference !!!')