"""在线模式入口：python main.py
回放模式入口：python main.py --replay videos/xxx.mp4
"""
import argparse
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")

def build_pipelines():
    """组装各摄像头的真实组件。海康抓图函数见 hik_adapter.py（用户接线点）。"""
    raise NotImplementedError("请先实现 hik_adapter.py 并在此组装 Pipeline")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", help="离线回放视频路径")
    args = parser.parse_args()
    if args.replay:
        from outlet_detect.replay_tool import replay
        replay(args.replay)
    else:
        from outlet_detect.scheduler import run_loop
        run_loop(build_pipelines())

if __name__ == "__main__":
    main()
